from __future__ import annotations

import numpy as np

Array = np.ndarray


class Layer:
    def forward(self, x: Array, training: bool = True) -> Array:
        raise NotImplementedError

    def backward(self, grad: Array) -> Array:
        raise NotImplementedError

    def parameters(self) -> list[tuple[Array, Array]]:
        return []


class Dense(Layer):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        rng: np.random.Generator,
        init: str = "he",
    ) -> None:
        if init == "he":
            scale = np.sqrt(2.0 / n_in)
        elif init == "xavier":
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
        x_2d = self._x.reshape(-1, self._x.shape[-1])
        g_2d = grad.reshape(-1, grad.shape[-1])
        self.dW[...] = x_2d.T @ g_2d
        self.db[...] = g_2d.sum(axis=0)
        return grad @ self.W.T

    def parameters(self) -> list[tuple[Array, Array]]:
        return [(self.W, self.dW), (self.b, self.db)]


class LayerNorm(Layer):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.dgamma = np.zeros(dim)
        self.dbeta = np.zeros(dim)
        self._x_hat: Array | None = None
        self._inv_std: Array | None = None

    def forward(self, x: Array, training: bool = True) -> Array:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._inv_std = 1.0 / np.sqrt(var + self.eps)
        self._x_hat = (x - mean) * self._inv_std
        return self.gamma * self._x_hat + self.beta

    def backward(self, grad: Array) -> Array:
        if self._x_hat is None or self._inv_std is None:
            raise RuntimeError("backward() called before forward()")
        axes = tuple(range(grad.ndim - 1))
        self.dgamma[...] = (grad * self._x_hat).sum(axis=axes)
        self.dbeta[...] = grad.sum(axis=axes)
        dx_hat = grad * self.gamma
        n = grad.shape[-1]
        return (
            self._inv_std
            / n
            * (
                n * dx_hat
                - dx_hat.sum(axis=-1, keepdims=True)
                - self._x_hat * (dx_hat * self._x_hat).sum(axis=-1, keepdims=True)
            )
        )

    def parameters(self) -> list[tuple[Array, Array]]:
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class Dropout(Layer):
    def __init__(self, rate: float = 0.1, seed: int = 0) -> None:
        if not 0.0 <= rate < 1.0:
            raise ValueError("dropout rate must be in [0, 1)")
        self.rate = rate
        self.rng = np.random.default_rng(seed)
        self._mask: Array | None = None

    def forward(self, x: Array, training: bool = True) -> Array:
        if not training or self.rate == 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.rate
        self._mask = (self.rng.random(x.shape) < keep) / keep
        return x * self._mask

    def backward(self, grad: Array) -> Array:
        return grad if self._mask is None else grad * self._mask


class ReLU(Layer):
    def __init__(self) -> None:
        self._mask: Array | None = None

    def forward(self, x: Array, training: bool = True) -> Array:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: Array) -> Array:
        return grad * self._mask


class GELU(Layer):
    def __init__(self) -> None:
        self._x: Array | None = None

    def forward(self, x: Array, training: bool = True) -> Array:
        self._x = x
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))

    def backward(self, grad: Array) -> Array:
        x = self._x
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x**3)
        th = np.tanh(inner)
        sech2 = 1.0 - th**2
        d_inner = c * (1.0 + 3.0 * 0.044715 * x**2)
        return grad * (0.5 * (1.0 + th) + 0.5 * x * sech2 * d_inner)


class Tanh(Layer):
    def __init__(self) -> None:
        self._out: Array | None = None

    def forward(self, x: Array, training: bool = True) -> Array:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: Array) -> Array:
        return grad * (1.0 - self._out**2)


class Sigmoid(Layer):
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


class MultiHeadAttention(Layer):
    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator) -> None:
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = Dense(d_model, d_model, rng, init="xavier")
        self.k_proj = Dense(d_model, d_model, rng, init="xavier")
        self.v_proj = Dense(d_model, d_model, rng, init="xavier")
        self.out_proj = Dense(d_model, d_model, rng, init="xavier")
        self._cache: tuple[Array, Array, Array, Array] | None = None

    def _split(self, x: Array) -> Array:
        b, seq, _ = x.shape
        return x.reshape(b, seq, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge(self, x: Array) -> Array:
        b, _, seq, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, seq, self.d_model)

    def forward(self, x: Array, training: bool = True) -> Array:
        q = self._split(self.q_proj.forward(x, training))
        k = self._split(self.k_proj.forward(x, training))
        v = self._split(self.v_proj.forward(x, training))
        scale = 1.0 / np.sqrt(self.d_head)
        scores = (q @ k.swapaxes(-2, -1)) * scale
        shifted = scores - scores.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        attn = exp / exp.sum(axis=-1, keepdims=True)
        context = self._merge(attn @ v)
        self._cache = (q, k, v, attn)
        return self.out_proj.forward(context, training)

    def backward(self, grad: Array) -> Array:
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")
        q, k, v, attn = self._cache
        d_context = self._split(self.out_proj.backward(grad))
        dv = attn.swapaxes(-2, -1) @ d_context
        d_attn = d_context @ v.swapaxes(-2, -1)
        d_scores = attn * (d_attn - (d_attn * attn).sum(axis=-1, keepdims=True))
        d_scores *= 1.0 / np.sqrt(self.d_head)
        dq = d_scores @ k
        dk = d_scores.swapaxes(-2, -1) @ q
        dx_q = self.q_proj.backward(self._merge(dq))
        dx_k = self.k_proj.backward(self._merge(dk))
        dx_v = self.v_proj.backward(self._merge(dv))
        return dx_q + dx_k + dx_v

    def parameters(self) -> list[tuple[Array, Array]]:
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.out_proj.parameters()
        )


ACTIVATIONS: dict[str, type[Layer]] = {
    "relu": ReLU,
    "gelu": GELU,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
}
