from __future__ import annotations

import numpy as np

Array = np.ndarray


def softmax(logits: Array) -> Array:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def l2_normalize(x: Array, eps: float = 1e-12) -> tuple[Array, Array]:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    safe_norms = np.maximum(norms, eps)
    return x / safe_norms, safe_norms


def l2_normalize_backward(grad_normed: Array, normed: Array, norms: Array) -> Array:
    proj = (grad_normed * normed).sum(axis=-1, keepdims=True)
    return (grad_normed - normed * proj) / norms


class SoftmaxCrossEntropy:
    def __init__(self) -> None:
        self._probs: Array | None = None
        self._labels: Array | None = None

    def forward(self, logits: Array, labels: Array) -> float:
        self._probs = softmax(logits)
        self._labels = labels
        picked = self._probs[np.arange(len(labels)), labels]
        return float(-np.log(picked + 1e-12).mean())

    def backward(self) -> Array:
        if self._probs is None or self._labels is None:
            raise RuntimeError("backward() called before forward()")
        grad = self._probs.copy()
        grad[np.arange(len(self._labels)), self._labels] -= 1.0
        return grad / len(self._labels)


class InfoNCELoss:
    def __init__(self, temperature: float = 0.07) -> None:
        self.temperature = temperature
        self._cache: tuple[Array, Array, Array, Array, Array, Array, Array] | None = None

    def forward(
        self,
        image_features: Array,
        text_features: Array,
        labels: Array | None = None,
    ) -> float:
        n = len(image_features)
        img_norm, img_len = l2_normalize(image_features)
        txt_norm, txt_len = l2_normalize(text_features)
        logits = (img_norm @ txt_norm.T) / self.temperature

        if labels is None:
            targets = np.eye(n)
        else:
            lbl = np.asarray(labels)
            match = (lbl[:, None] == lbl[None, :]).astype(float)
            targets = match / np.maximum(match.sum(axis=1, keepdims=True), 1.0)

        probs_i2t = softmax(logits)
        probs_t2i = softmax(logits.T)
        loss_i2t = -np.sum(targets * np.log(probs_i2t + 1e-12), axis=1).mean()
        loss_t2i = -np.sum(targets.T * np.log(probs_t2i + 1e-12), axis=1).mean()
        self._cache = (img_norm, img_len, txt_norm, txt_len, probs_i2t, probs_t2i, targets)
        return float(0.5 * (loss_i2t + loss_t2i))

    def backward(self) -> tuple[Array, Array]:
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        img_norm, img_len, txt_norm, txt_len, probs_i2t, probs_t2i, targets = self._cache
        n = len(img_norm)
        scale = 0.5 / (n * self.temperature)
        d_logits = ((probs_i2t - targets) + (probs_t2i - targets.T).T) * scale
        d_img_norm = d_logits @ txt_norm
        d_txt_norm = d_logits.T @ img_norm
        d_img = l2_normalize_backward(d_img_norm, img_norm, img_len)
        d_txt = l2_normalize_backward(d_txt_norm, txt_norm, txt_len)
        return d_img, d_txt
