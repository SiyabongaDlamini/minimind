"""Built-in example datasets (require the optional ``demo`` extra)."""

from __future__ import annotations

from .data import Dataset


def load_digits() -> Dataset:
    """The 8x8 handwritten digits dataset (1,797 samples), scaled to ``[0, 1]``."""
    try:
        from sklearn.datasets import load_digits as _load
    except ImportError as exc:  # pragma: no cover
        raise ImportError("load_digits needs scikit-learn: pip install 'minimind[demo]'") from exc
    X, y = _load(return_X_y=True)
    return Dataset(X / 16.0, y, tuple(str(i) for i in range(10)), image_shape=(8, 8))

