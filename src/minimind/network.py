from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .layers import ACTIVATIONS, Dense, Dropout, Layer, LayerNorm
from .losses import SoftmaxCrossEntropy, softmax
from .optim import LRScheduler, Optimizer, clip_grad_norm, make_lr_scheduler, make_optimizer

Array = np.ndarray
History = dict[str, list[float]]
Callback = Callable[[int, int, "NeuralNetwork", History], "bool | None"]


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "relu",
        seed: int = 0,
        dropout: float = 0.0,
        layer_norm: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        sizes = tuple(int(s) for s in layer_sizes)
        if len(sizes) < 2 or any(s < 1 for s in sizes):
            raise ValueError("layer_sizes needs at least input and output sizes, all >= 1")
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}"
            )
        self.layer_sizes = sizes
        self.activation = activation
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.label_smoothing = label_smoothing
        self.meta: dict[str, Any] = {}
        self._loss = SoftmaxCrossEntropy(label_smoothing=label_smoothing)

        rng = np.random.default_rng(seed)
        init = "he" if activation in ("relu", "gelu", "silu", "leaky_relu") else "xavier"
        self.layers: list[Layer] = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            self.layers.append(Dense(n_in, n_out, rng, init))
            if i < len(sizes) - 2:
                if layer_norm:
                    self.layers.append(LayerNorm(n_out))
                self.layers.append(ACTIVATIONS[activation]())
                if dropout > 0.0:
                    self.layers.append(Dropout(dropout, seed=seed + i + 1))

    def forward(self, X: Array, training: bool = False) -> Array:
        out = self._check_X(X)
        for layer in self.layers:
            out = layer.forward(out, training)
        return out

    def predict_proba(self, X: Array) -> Array:
        return softmax(self.forward(X))

    def predict(self, X: Array) -> Array:
        return self.forward(X).argmax(axis=1)

    def evaluate(self, X: Array, y: Array) -> tuple[float, float]:
        y = self._check_y(y, len(X))
        logits = self.forward(X)
        loss = SoftmaxCrossEntropy().forward(logits, y)
        return loss, float((logits.argmax(axis=1) == y).mean())

    def forward_trace(self, x: Array) -> list[Array]:
        out = self._check_X(np.atleast_2d(x))
        trace = [out]
        for layer in self.layers:
            out = layer.forward(out, training=False)
            if isinstance(layer, tuple(ACTIVATIONS.values())):
                trace.append(out)
        trace.append(softmax(out))
        return trace

    def fit(
        self,
        X: Array,
        y: Array,
        *,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer: str | Optimizer = "adam",
        lr: float = 0.01,
        lr_schedule: str | LRScheduler | None = None,
        max_grad_norm: float = 0.0,
        label_smoothing: float | None = None,
        validation: tuple[Array, Array] | None = None,
        patience: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        callbacks: Sequence[Callback] = (),
        verbose: int = 0,
    ) -> History:
        X = self._check_X(X)
        y = self._check_y(y, len(X))
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        if patience < 0:
            raise ValueError("patience must be >= 0")
        opt = make_optimizer(optimizer, lr)
        base_lr = getattr(opt, "lr", lr)
        scheduler = make_lr_scheduler(lr_schedule, base_lr, epochs)
        loss_fn = (
            self._loss
            if label_smoothing is None
            else SoftmaxCrossEntropy(label_smoothing=label_smoothing)
        )
        params = [pg for layer in self.layers for pg in layer.parameters()]
        rng = np.random.default_rng(seed)

        history: History = {"loss": [], "accuracy": []}
        if validation is not None:
            history.update(val_loss=[], val_accuracy=[])
        if scheduler is not None:
            history["lr"] = []
        for cb in callbacks:
            cb(0, epochs, self, history)

        best_val_loss = float("inf")
        best_params: list[Array] | None = None
        wait = 0

        n = len(X)
        for epoch in range(1, epochs + 1):
            if scheduler is not None:
                cur_lr = scheduler.step(opt, epoch, epochs)
                history["lr"].append(cur_lr)
            order = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                logits = self.forward(X[idx], training=True)
                loss_fn.forward(logits, y[idx])
                grad = loss_fn.backward()
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                if max_grad_norm > 0.0:
                    clip_grad_norm(params, max_grad_norm)
                opt.step(params)

            loss, acc = self.evaluate(X, y)
            history["loss"].append(loss)
            history["accuracy"].append(acc)
            if validation is not None:
                vloss, vacc = self.evaluate(*validation)
                history["val_loss"].append(vloss)
                history["val_accuracy"].append(vacc)
                if patience > 0:
                    if vloss < best_val_loss - 1e-8:
                        best_val_loss = vloss
                        best_params = [p.copy() for p, _ in params]
                        wait = 0
                    else:
                        wait += 1
            if verbose and (epoch % verbose == 0 or epoch == epochs):
                line = f"epoch {epoch:>4}/{epochs}  loss {loss:.4f}  acc {acc:.3f}"
                if validation is not None:
                    line += f"  val_loss {vloss:.4f}  val_acc {vacc:.3f}"
                print(line)
            if any(cb(epoch, epochs, self, history) for cb in callbacks):
                break
            if patience > 0 and validation is not None and wait >= patience:
                break

        if best_params is not None:
            for (p, _), saved in zip(params, best_params, strict=True):
                p[...] = saved
        return history

    def get_weights(self) -> list[tuple[Array, Array]]:
        return [(d.W.copy(), d.b.copy()) for d in self._dense_layers()]

    def set_weights(self, weights: Sequence[tuple[Array, Array]]) -> None:
        dense = self._dense_layers()
        if len(weights) != len(dense):
            raise ValueError("weight list does not match the number of dense layers")
        for layer, (W, b) in zip(dense, weights, strict=True):
            if W.shape != layer.W.shape or b.shape != layer.b.shape:
                raise ValueError("weight shapes do not match this network")
            layer.W[...] = W
            layer.b[...] = b

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        arrays: dict[str, Array] = {
            "layer_sizes": np.array(self.layer_sizes),
            "activation": np.array(self.activation),
            "dropout": np.array(self.dropout, dtype=float),
            "layer_norm": np.array(self.layer_norm, dtype=bool),
            "label_smoothing": np.array(self.label_smoothing, dtype=float),
            "meta": np.array(json.dumps(meta if meta is not None else self.meta)),
        }
        for i, (W, b) in enumerate(self.get_weights()):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        for i, ln in enumerate(self._layer_norm_layers()):
            arrays[f"LN_gamma{i}"] = ln.gamma.copy()
            arrays[f"LN_beta{i}"] = ln.beta.copy()
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> NeuralNetwork:
        with np.load(Path(path), allow_pickle=False) as data:
            dropout = float(data["dropout"]) if "dropout" in data else 0.0
            layer_norm = bool(data["layer_norm"]) if "layer_norm" in data else False
            label_smoothing = (
                float(data["label_smoothing"]) if "label_smoothing" in data else 0.0
            )
            net = cls(
                data["layer_sizes"].tolist(),
                activation=str(data["activation"]),
                dropout=dropout,
                layer_norm=layer_norm,
                label_smoothing=label_smoothing,
            )
            n_dense = len(net._dense_layers())
            net.set_weights([(data[f"W{i}"], data[f"b{i}"]) for i in range(n_dense)])
            for i, ln in enumerate(net._layer_norm_layers()):
                if f"LN_gamma{i}" in data and f"LN_beta{i}" in data:
                    ln.gamma[...] = data[f"LN_gamma{i}"]
                    ln.beta[...] = data[f"LN_beta{i}"]
            net.meta = json.loads(str(data["meta"]))
        return net

    def _dense_layers(self) -> list[Dense]:
        return [layer for layer in self.layers if isinstance(layer, Dense)]

    def _layer_norm_layers(self) -> list[LayerNorm]:
        return [layer for layer in self.layers if isinstance(layer, LayerNorm)]

    def _check_X(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"expected X of shape (n, {self.layer_sizes[0]}), got {X.shape}"
            )
        return X

    def _check_y(self, y: Array, n: int) -> Array:
        y = np.asarray(y)
        if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError(f"expected y as {n} integer class labels, got shape {y.shape}")
        if y.min() < 0 or y.max() >= self.layer_sizes[-1]:
            raise ValueError(f"labels must be in [0, {self.layer_sizes[-1] - 1}]")
        return y

    def __repr__(self) -> str:
        arch = "-".join(str(s) for s in self.layer_sizes)
        return f"NeuralNetwork({arch}, activation={self.activation!r})"
