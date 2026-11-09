"""Dataset container and small preprocessing helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class Dataset:
    """Features, integer labels and the human-readable class names.

    Attributes:
        X: Feature matrix ``(n_samples, n_features)``.
        y: Integer labels ``(n_samples,)`` indexing into ``class_names``.
        class_names: Name of each class.
        image_shape: ``(height, width)`` if each row is a flattened image, else ``None``.
    """

    X: Array
    y: Array
    class_names: tuple[str, ...]
    image_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=int)
        if self.X.ndim != 2 or self.y.shape != (len(self.X),):
            raise ValueError("X must be 2-D and y must have one label per row")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def split(
        self, test_size: float = 0.2, seed: int = 0
    ) -> tuple[Dataset, Dataset]:
        """Stratified train/test split that keeps class proportions."""
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1")
        rng = np.random.default_rng(seed)
        test_idx: list[int] = []
        for cls in np.unique(self.y):
            members = rng.permutation(np.flatnonzero(self.y == cls))
            n_test = min(max(1, math.ceil(test_size * len(members))), len(members) - 1)
            test_idx.extend(members[: max(n_test, 0)].tolist())
        mask = np.zeros(len(self), dtype=bool)
        mask[test_idx] = True
        return self._take(~mask), self._take(mask)

    def _take(self, mask: Array) -> Dataset:
        return Dataset(self.X[mask], self.y[mask], self.class_names, self.image_shape)


class Standardizer:
    """Scale each feature to zero mean and unit variance using training statistics."""

    def __init__(self) -> None:
        self.mean: Array | None = None
        self.std: Array | None = None

    def fit(self, X: Array) -> Standardizer:
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, X: Array) -> Array:
        if self.mean is None or self.std is None:
            raise RuntimeError("call fit() before transform()")
        return (np.asarray(X, dtype=float) - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        if self.mean is None or self.std is None:
            raise RuntimeError("call fit() first")
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> Standardizer:
        s = cls()
        s.mean, s.std = np.array(d["mean"]), np.array(d["std"])
        return s
