from __future__ import annotations

import numpy as np

Array = np.ndarray
Params = list[tuple[Array, Array]]


def clip_grad_norm(params: Params, max_norm: float) -> float:
    total_sq = sum(float(np.sum(g * g)) for _, g in params)
    norm = float(np.sqrt(total_sq))
    if norm > max_norm > 0:
        scale = max_norm / (norm + 1e-12)
        for _, g in params:
            g *= scale
    return norm


class Optimizer:
    def step(self, params: Params) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.1, momentum: float = 0.0, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: dict[int, Array] = {}

    def step(self, params: Params) -> None:
        for i, (p, g) in enumerate(params):
            grad = g + self.weight_decay * p if self.weight_decay and p.ndim >= 2 else g
            if self.momentum:
                v = self._velocity.setdefault(i, np.zeros_like(p))
                v *= self.momentum
                v -= self.lr * grad
                p += v
            else:
                p -= self.lr * grad


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.weight_decay = weight_decay
        self._m: dict[int, Array] = {}
        self._v: dict[int, Array] = {}
        self._t = 0

    def step(self, params: Params) -> None:
        self._t += 1
        c1 = 1.0 - self.beta1**self._t
        c2 = 1.0 - self.beta2**self._t
        for i, (p, g) in enumerate(params):
            if self.weight_decay and p.ndim >= 2:
                p *= 1.0 - self.lr * self.weight_decay
            m = self._m.setdefault(i, np.zeros_like(p))
            v = self._v.setdefault(i, np.zeros_like(p))
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


class AdamW(Adam):
    def __init__(
        self,
        lr: float = 0.003,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)


def make_optimizer(spec: str | Optimizer, lr: float) -> Optimizer:
    if isinstance(spec, Optimizer):
        return spec
    name = spec.lower()
    if name == "sgd":
        return SGD(lr=lr)
    if name == "adam":
        return Adam(lr=lr)
    if name == "adamw":
        return AdamW(lr=lr)
    raise ValueError(f"unknown optimizer {spec!r}; expected 'sgd', 'adam', or 'adamw'")
