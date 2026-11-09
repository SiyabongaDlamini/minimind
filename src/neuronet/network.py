"""A small multilayer perceptron classifier implemented in pure NumPy."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .layers import ACTIVATIONS, Dense, Layer
from .losses import SoftmaxCrossEntropy, softmax
from .optim import Optimizer, make_optimizer

Array = np.ndarray
History = dict[str, list[float]]
# callback(epoch, total_epochs, network, history) -> True to stop training early.
# It is called once with epoch 0 (before any training) and then after every epoch.
Callback = Callable[[int, int, "NeuralNetwork", History], "bool | None"]


class NeuralNetwork:
    """Fully connected softmax classifier.

    Args:
        layer_sizes: Sizes from input to output, e.g. ``[64, 32, 10]``.
        activation: Hidden activation: ``'relu'``, ``'tanh'`` or ``'sigmoid'``.
        seed: Seed for weight initialisation.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activation: str = "relu",
        seed: int = 0,
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
        self.meta: dict[str, Any] = {}
        self._loss = SoftmaxCrossEntropy()

        rng = np.random.default_rng(seed)
        init = "he" if activation == "relu" else "xavier"
        self.layers: list[Layer] = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            self.layers.append(Dense(n_in, n_out, rng, init))
            if i < len(sizes) - 2:
                self.layers.append(ACTIVATIONS[activation]())

    # ------------------------------------------------------------------ inference
    def forward(self, X: Array, training: bool = False) -> Array:
        """Return the raw logits for ``X`` with shape ``(n_samples, n_classes)``."""
        out = self._check_X(X)
        for layer in self.layers:
            out = layer.forward(out, training)
        return out

    def predict_proba(self, X: Array) -> Array:
        """Class probabilities for each row of ``X``."""
        return softmax(self.forward(X))

    def predict(self, X: Array) -> Array:
        """Most likely class index for each row of ``X``."""
        return self.forward(X).argmax(axis=1)

    def evaluate(self, X: Array, y: Array) -> tuple[float, float]:
        """Return ``(loss, accuracy)`` on ``(X, y)``."""
        y = self._check_y(y, len(X))
        logits = self.forward(X)
        loss = SoftmaxCrossEntropy().forward(logits, y)
        return loss, float((logits.argmax(axis=1) == y).mean())

    def forward_trace(self, x: Array) -> list[Array]:
        """Activations at every layer boundary for one sample (used by the visualiser).

        Returns ``[input, hidden_1, ..., hidden_n, probabilities]``, each of shape
        ``(1, size)``.
        """
        out = self._check_X(np.atleast_2d(x))
        trace = [out]
        for layer in self.layers:
            out = layer.forward(out, training=False)
            if not isinstance(layer, Dense):
                trace.append(out)
        trace.append(softmax(out))
        return trace

    # ------------------------------------------------------------------- training
    def fit(
        self,
        X: Array,
        y: Array,
        *,
        epochs: int = 100,
        batch_size: int = 32,
        optimizer: str | Optimizer = "adam",
        lr: float = 0.01,
        validation: tuple[Array, Array] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        callbacks: Sequence[Callback] = (),
        verbose: int = 0,
    ) -> History:
        """Train with mini-batch gradient descent and return the metric history.

        Args:
            X: Features, shape ``(n_samples, n_features)``.
            y: Integer class labels, shape ``(n_samples,)``.
            epochs: Number of passes over the data.
            batch_size: Mini-batch size.
            optimizer: ``'adam'``, ``'sgd'`` or an :class:`~neuronet.optim.Optimizer`.
            lr: Learning rate when ``optimizer`` is given by name.
            validation: Optional ``(X_val, y_val)`` evaluated after every epoch.
            shuffle: Reshuffle the training data every epoch.
            seed: Seed for shuffling.
            callbacks: Functions called after each epoch; see :data:`Callback`.
            verbose: Print a progress line every ``verbose`` epochs (0 = silent).
        """
        X = self._check_X(X)
        y = self._check_y(y, len(X))
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        opt = make_optimizer(optimizer, lr)
        params = [pg for layer in self.layers for pg in layer.parameters()]
        rng = np.random.default_rng(seed)

        history: History = {"loss": [], "accuracy": []}
        if validation is not None:
            history.update(val_loss=[], val_accuracy=[])
        for cb in callbacks:
            cb(0, epochs, self, history)

        n = len(X)
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                logits = self.forward(X[idx], training=True)
                self._loss.forward(logits, y[idx])
                grad = self._loss.backward()
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                opt.step(params)

            loss, acc = self.evaluate(X, y)
            history["loss"].append(loss)
            history["accuracy"].append(acc)
            if validation is not None:
                vloss, vacc = self.evaluate(*validation)
                history["val_loss"].append(vloss)
                history["val_accuracy"].append(vacc)
            if verbose and (epoch % verbose == 0 or epoch == epochs):
                line = f"epoch {epoch:>4}/{epochs}  loss {loss:.4f}  acc {acc:.3f}"
                if validation is not None:
                    line += f"  val_loss {vloss:.4f}  val_acc {vacc:.3f}"
                print(line)
            if any(cb(epoch, epochs, self, history) for cb in callbacks):
                break
        return history

    # ---------------------------------------------------------------- persistence
    def get_weights(self) -> list[tuple[Array, Array]]:
        """Copies of ``(W, b)`` for every dense layer."""
        return [(d.W.copy(), d.b.copy()) for d in self._dense_layers()]

    def set_weights(self, weights: Sequence[tuple[Array, Array]]) -> None:
        """Load weights produced by :meth:`get_weights`."""
        dense = self._dense_layers()
        if len(weights) != len(dense):
            raise ValueError("weight list does not match the number of dense layers")
        for layer, (W, b) in zip(dense, weights, strict=True):
            if W.shape != layer.W.shape or b.shape != layer.b.shape:
                raise ValueError("weight shapes do not match this network")
            layer.W[...] = W
            layer.b[...] = b

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        """Save architecture, weights and optional JSON-serialisable ``meta`` to ``.npz``."""
        arrays: dict[str, Array] = {
            "layer_sizes": np.array(self.layer_sizes),
            "activation": np.array(self.activation),
            "meta": np.array(json.dumps(meta if meta is not None else self.meta)),
        }
        for i, (W, b) in enumerate(self.get_weights()):
            arrays[f"W{i}"], arrays[f"b{i}"] = W, b
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> NeuralNetwork:
        """Load a network written by :meth:`save` (no pickle is used)."""
        with np.load(Path(path), allow_pickle=False) as data:
            net = cls(data["layer_sizes"].tolist(), activation=str(data["activation"]))
            n_dense = len(net._dense_layers())
            net.set_weights([(data[f"W{i}"], data[f"b{i}"]) for i in range(n_dense)])
            net.meta = json.loads(str(data["meta"]))
        return net

    # -------------------------------------------------------------------- helpers
    def _dense_layers(self) -> list[Dense]:
        return [layer for layer in self.layers if isinstance(layer, Dense)]

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
