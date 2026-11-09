from __future__ import annotations

from .data import Dataset


def load_digits() -> Dataset:
    try:
        from sklearn.datasets import load_digits as _load
    except ImportError as exc:
        raise ImportError("load_digits needs scikit-learn: pip install 'minimind[demo]'") from exc
    X, y = _load(return_X_y=True)
    return Dataset(X / 16.0, y, tuple(str(i) for i in range(10)), image_shape=(8, 8))
