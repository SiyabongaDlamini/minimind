"""Parsers that turn images and tabular files into :class:`~minimind.data.Dataset` rows.

The image parser is built for scans and photos of documents: it converts to
grayscale, flips dark-ink-on-light-paper into the bright-ink convention used by
datasets like MNIST, stretches contrast, crops to the content and resizes.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import Dataset

Array = np.ndarray
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("image parsing needs Pillow: pip install 'minimind[image]'") from exc
    return Image


@dataclass(frozen=True)
class ImageParser:
    """Convert an image into a flat feature vector in ``[0, 1]``.

    Args:
        size: Output ``(height, width)``.
        invert: ``True`` always inverts, ``False`` never, ``None`` detects it from the
            border brightness (light background -> invert so ink becomes bright).
        crop: Crop to the bounding box of the content before resizing.
        margin: Padding around the cropped content, as a fraction of its size.
    """

    size: tuple[int, int] = (8, 8)
    invert: bool | None = None
    crop: bool = True
    margin: float = 0.1

    def parse(self, source: str | Path | Array | Any) -> Array:
        """Parse a file path, ``PIL.Image`` or ``numpy`` array into a vector."""
        gray = self._to_gray(source)
        if self._should_invert(gray):
            gray = 1.0 - gray
        span = gray.max() - gray.min()
        gray = (gray - gray.min()) / span if span > 1e-9 else np.zeros_like(gray)
        if self.crop:
            gray = self._crop(gray)
        return self._resize(gray).ravel()

    def parse_many(self, sources: Iterable[Any]) -> Array:
        """Parse several sources into an ``(n, height * width)`` matrix."""
        rows = [self.parse(s) for s in sources]
        if not rows:
            raise ValueError("no images to parse")
        return np.stack(rows)

    # ---------------------------------------------------------------- internals
    def _to_gray(self, source: Any) -> Array:
        Image = _pil()
        if isinstance(source, (str, Path)):
            img = Image.open(source)
        elif isinstance(source, np.ndarray):
            arr = source
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        else:
            img = source
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba)
        return np.asarray(img.convert("L"), dtype=float) / 255.0

    def _should_invert(self, gray: Array) -> bool:
        if self.invert is not None:
            return self.invert
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        return bool(np.median(border) > 0.5)

    def _crop(self, gray: Array, threshold: float = 0.25) -> Array:
        rows, cols = np.where(gray > threshold)
        if rows.size == 0:
            return gray
        crop = gray[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        h, w = crop.shape
        target_ratio = self.size[1] / self.size[0]
        new_h, new_w = (h, math.ceil(h * target_ratio)) if w / h < target_ratio else (
            math.ceil(w / target_ratio),
            w,
        )
        pad_h, pad_w = round(self.margin * new_h), round(self.margin * new_w)
        out = np.zeros((new_h + 2 * pad_h, new_w + 2 * pad_w))
        top, left = pad_h + (new_h - h) // 2, pad_w + (new_w - w) // 2
        out[top : top + h, left : left + w] = crop
        return out

    def _resize(self, gray: Array) -> Array:
        Image = _pil()
        img = Image.fromarray((gray * 255).astype(np.uint8))
        img = img.resize((self.size[1], self.size[0]), Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=float) / 255.0


def load_image_folder(
    root: str | Path,
    parser: ImageParser | None = None,
) -> Dataset:
    """Load ``root/<class_name>/*.png`` style folders; each subfolder is one class."""
    root = Path(root)
    parser = parser or ImageParser()
    if not root.is_dir():
        raise FileNotFoundError(f"image folder not found: {root}")
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(class_dirs) < 2:
        raise ValueError(f"{root} needs at least two class subfolders, found {len(class_dirs)}")
    rows: list[Array] = []
    labels: list[int] = []
    for label, folder in enumerate(class_dirs):
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(parser.parse(path))
                labels.append(label)
    if not rows:
        raise ValueError(f"no images with extensions {IMAGE_EXTENSIONS} found under {root}")
    return Dataset(
        np.stack(rows),
        np.array(labels),
        tuple(p.name for p in class_dirs),
        image_shape=parser.size,
    )


def load_csv(
    path: str | Path,
    target: str | int = -1,
    delimiter: str = ",",
) -> Dataset:
    """Load a CSV with a header row; ``target`` is a column name or index (default: last).

    Feature columns must be numeric. Labels may be text; they are mapped to integer
    classes in sorted order.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh, delimiter=delimiter) if r]
    if len(rows) < 2:
        raise ValueError(f"{path} needs a header row and at least one data row")
    header, body = rows[0], rows[1:]
    t = header.index(target) if isinstance(target, str) else target % len(header)
    feature_cols = [i for i in range(len(header)) if i != t]
    try:
        X = np.array([[float(r[i]) for i in feature_cols] for r in body])
    except ValueError as exc:
        raise ValueError(f"non-numeric feature value in {path}: {exc}") from exc
    names = sorted({r[t].strip() for r in body})
    lookup = {n: i for i, n in enumerate(names)}
    y = np.array([lookup[r[t].strip()] for r in body])
    return Dataset(X, y, tuple(names))
