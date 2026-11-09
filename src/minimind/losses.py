"""Loss functions."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def softmax(logits: Array) -> Array:
    """Row-wise softmax, shifted by the row max for numerical stability."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Softmax followed by cross-entropy, fused so the gradient is simply ``p - y``."""

    def __init__(self) -> None:
        self._probs: Array | None = None
        self._labels: Array | None = None

    def forward(self, logits: Array, labels: Array) -> float:
        """Mean cross-entropy of ``logits`` against integer class ``labels``."""
        self._probs = softmax(logits)
        self._labels = labels
        picked = self._probs[np.arange(len(labels)), labels]
        return float(-np.log(picked + 1e-12).mean())

    def backward(self) -> Array:
        """Gradient of the mean loss with respect to the logits."""
        if self._probs is None or self._labels is None:
            raise RuntimeError("backward() called before forward()")
        grad = self._probs.copy()
        grad[np.arange(len(self._labels)), self._labels] -= 1.0
        return grad / len(self._labels)
