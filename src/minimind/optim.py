"""Gradient-descent optimizers."""

from __future__ import annotations

import numpy as np

Array = np.ndarray
Params = list[tuple[Array, Array]]


class Optimizer:
    """Base class. ``step`` updates parameter arrays in place."""

    def step(self, params: Params) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum."""

    def __init__(self, lr: float = 0.1, momentum: float = 0.0) -> None:
        self.lr = lr
        self.momentum = momentum
        self._velocity: dict[int, Array] = {}

    def step(self, params: Params) -> None:
        for i, (p, g) in enumerate(params):
            if self.momentum:
                v = self._velocity.setdefault(i, np.zeros_like(p))
                v *= self.momentum
                v -= self.lr * g
                p += v
            else:
                p -= self.lr * g


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015)."""

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self._m: dict[int, Array] = {}
        self._v: dict[int, Array] = {}
        self._t = 0

    def step(self, params: Params) -> None:
        self._t += 1
        correction1 = 1.0 - self.beta1**self._t
        correction2 = 1.0 - self.beta2**self._t
        for i, (p, g) in enumerate(params):
            m = self._m.setdefault(i, np.zeros_like(p))
            v = self._v.setdefault(i, np.zeros_like(p))
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= self.lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)


def make_optimizer(spec: str | Optimizer, lr: float) -> Optimizer:
    """Build an optimizer from a name (``'sgd'`` or ``'adam'``) or pass one through."""
    if isinstance(spec, Optimizer):
        return spec
    name = spec.lower()
    if name == "sgd":
        return SGD(lr=lr)
    if name == "adam":
        return Adam(lr=lr)
    raise ValueError(f"unknown optimizer {spec!r}; expected 'sgd' or 'adam'")
