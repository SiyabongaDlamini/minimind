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


class LRScheduler:
    def get_lr(self, epoch: int, total_epochs: int) -> float:
        raise NotImplementedError

    def step(self, optimizer: Optimizer, epoch: int, total_epochs: int) -> float:
        lr = self.get_lr(epoch, total_epochs)
        optimizer.lr = lr  # type: ignore[attr-defined]
        return lr


class CosineAnnealingLR(LRScheduler):
    def __init__(self, base_lr: float, min_lr: float = 0.0) -> None:
        self.base_lr = base_lr
        self.min_lr = min_lr

    def get_lr(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return self.base_lr
        progress = min(max(epoch - 1, 0) / (total_epochs - 1), 1.0)
        return float(
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + np.cos(np.pi * progress))
        )


class StepLR(LRScheduler):
    def __init__(self, base_lr: float, step_size: int = 20, gamma: float = 0.5) -> None:
        if step_size < 1:
            raise ValueError("step_size must be >= 1")
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch: int, total_epochs: int) -> float:
        steps = max(epoch - 1, 0) // self.step_size
        return float(self.base_lr * (self.gamma**steps))


def make_lr_scheduler(
    spec: str | LRScheduler | None,
    base_lr: float,
    total_epochs: int,
) -> LRScheduler | None:
    if spec is None or isinstance(spec, LRScheduler):
        return spec
    name = spec.lower()
    if name == "cosine":
        return CosineAnnealingLR(base_lr=base_lr, min_lr=base_lr * 0.01)
    if name == "step":
        return StepLR(base_lr=base_lr, step_size=max(1, total_epochs // 3), gamma=0.5)
    raise ValueError(f"unknown lr_schedule {spec!r}; expected 'cosine', 'step', or None")
