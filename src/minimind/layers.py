"""Network layers: a fully connected layer and the usual activation functions.

Every layer implements ``forward`` and ``backward``. ``backward`` receives the
gradient of the loss with respect to the layer's output and returns the gradient
with respect to its input, storing parameter gradients on the layer as it goes.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


class Layer:
    """Base class for all layers."""

    def forward(self, x: Array, training: bool = True) -> Array:
        raise NotImplementedError

    def backward(self, grad: Array) -> Array:
        raise NotImplementedError

    def parameters(self) -> list[tuple[Array, Array]]:
        """Return ``(parameter, gradient)`` pairs; arrays are updated in place."""
        return []


class Dense(Layer):
    """Fully connected layer computing ``x @ W + b``."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        rng: np.random.Generator,
        init: str = "he",
    ) -> None:
        if init == "he":  # suited to ReLU
            scale = np.sqrt(2.0 / n_in)
        elif init == "xavier":  # suited to tanh / sigmoid
            scale = np.sqrt(2.0 / (n_in + n_out))
        else:
            raise ValueError(f"unknown init {init!r}; expected 'he' or 'xavier'")
        self.W = rng.standard_normal((n_in, n_out)) * scale
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: Array | None = None

    def forward(self, x: Array, training: bool = True) -> Array:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: Array) -> Array:
        if self._x is None:
            raise RuntimeError("backward() called before forward()")
        self.dW[...] = self._x.T @ grad
        self.db[...] = grad.sum(axis=0)
        return grad @ self.W.T

    def parameters(self) -> list[tuple[Array, Array]]:
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU(Layer):
    """Rectified linear unit: ``max(0, x)``."""

    def __init__(self) -> None:
        self._mask: Array | None = None

    def forward(self, x: Array, training: bool = True) -> Array:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: Array) -> Array:
        return grad * self._mask


class Tanh(Layer):
    """Hyperbolic tangent."""

    def __init__(self) -> None:
        self._out: Array | None = None

    def forward(self, x: Array, training: bool = True) -> Array:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: Array) -> Array:
        return grad * (1.0 - self._out**2)


class Sigmoid(Layer):
    """Logistic sigmoid, computed in a numerically stable way."""

    def __init__(self) -> None:
        self._out: Array | None = None

    def forward(self, x: Array, training: bool = True) -> Array:
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        self._out = out
        return out

    def backward(self, grad: Array) -> Array:
        return grad * self._out * (1.0 - self._out)


ACTIVATIONS: dict[str, type[Layer]] = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}
