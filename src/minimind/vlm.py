from __future__ import annotations

import json
import re
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .layers import GELU, Dense, Layer, LayerNorm
from .losses import InfoNCELoss, l2_normalize, softmax
from .optim import AdamW, clip_grad_norm, make_optimizer
from .parsers import IMAGE_EXTENSIONS, ImageParser

Array = np.ndarray
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stable_hash(text: str, buckets: int) -> int:
    return zlib.crc32(text.encode("utf-8")) % buckets


class VisionBackbone:
    def __init__(self, image_size: tuple[int, int] = (16, 16)) -> None:
        self.image_size = image_size
        self.parser = ImageParser(size=image_size)

    @property
    def out_dim(self) -> int:
        h, w = self.image_size
        return h * w + 32 + 16 + 8

    def extract_one(self, source: Any) -> Array:
        h, w = self.image_size
        if isinstance(source, np.ndarray) and source.ndim == 1 and source.size == h * w:
            img = source.reshape(h, w)
        else:
            img = self.parser.parse(source).reshape(h, w)

        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        gy[1:-1, :] = img[2:, :] - img[:-2, :]
        mag = np.hypot(gx, gy)
        angle = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)

        n_bins = 8
        bins = np.clip((angle * n_bins).astype(int), 0, n_bins - 1)
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            (slice(0, mid_h), slice(0, mid_w)),
            (slice(0, mid_h), slice(mid_w, w)),
            (slice(mid_h, h), slice(0, mid_w)),
            (slice(mid_h, h), slice(mid_w, w)),
        ]
        hog = np.zeros(4 * n_bins)
        for q_idx, (rs, cs) in enumerate(quadrants):
            q_mag = mag[rs, cs].ravel()
            q_bin = bins[rs, cs].ravel()
            for b in range(n_bins):
                hog[q_idx * n_bins + b] = q_mag[q_bin == b].sum()
        hog_norm = np.linalg.norm(hog)
        if hog_norm > 1e-9:
            hog /= hog_norm

        grid_4x4 = np.zeros(16)
        sh, sw = max(1, h // 4), max(1, w // 4)
        for r in range(4):
            for c in range(4):
                patch = img[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw]
                grid_4x4[r * 4 + c] = patch.mean() if patch.size else 0.0

        yy, xx = np.ogrid[:h, :w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        dist = np.hypot(yy - cy, xx - cx) / max(np.hypot(cy, cx), 1.0)
        radial = np.zeros(8)
        for i in range(4):
            mask = (dist >= i * 0.25) & (dist < (i + 1) * 0.25 + 1e-6)
            vals = img[mask]
            radial[2 * i] = vals.mean() if vals.size else 0.0
            radial[2 * i + 1] = vals.std() if vals.size else 0.0

        return np.concatenate([img.ravel(), hog, grid_4x4, radial])

    def extract_batch(self, sources: Iterable[Any]) -> Array:
        rows = [self.extract_one(s) for s in sources]
        if not rows:
            raise ValueError("no images provided")
        return np.stack(rows)


class TextBackbone:
    def __init__(self, vocab_buckets: int = 512) -> None:
        self.vocab_buckets = vocab_buckets

    @property
    def out_dim(self) -> int:
        return self.vocab_buckets

    def extract_one(self, text: str) -> Array:
        vec = np.zeros(self.vocab_buckets, dtype=float)
        words = _TOKEN_RE.findall(text.lower())
        if not words:
            return vec

        for w in words:
            vec[_stable_hash(f"w:{w}", self.vocab_buckets)] += 2.0
            if len(w) > 3 and w.endswith("s"):
                vec[_stable_hash(f"w:{w[:-1]}", self.vocab_buckets)] += 1.5
            padded = f"<{w}>"
            for n in (3, 4):
                for i in range(max(0, len(padded) - n + 1)):
                    gram = padded[i : i + n]
                    vec[_stable_hash(f"g:{gram}", self.vocab_buckets)] += 0.5

        for w1, w2 in zip(words[:-1], words[1:], strict=False):
            vec[_stable_hash(f"b:{w1}_{w2}", self.vocab_buckets)] += 1.2

        vec = np.log1p(vec)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-9 else vec

    def extract_batch(self, texts: Sequence[str]) -> Array:
        if isinstance(texts, str):
            texts = [texts]
        rows = [self.extract_one(t) for t in texts]
        if not rows:
            raise ValueError("no texts provided")
        return np.stack(rows)


class ProjectionTower:
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embed_dim: int,
        rng: np.random.Generator,
    ) -> None:
        self.layers: list[Layer] = [
            Dense(in_dim, hidden_dim, rng, init="he"),
            LayerNorm(hidden_dim),
            GELU(),
            Dense(hidden_dim, embed_dim, rng, init="xavier"),
        ]

    def forward(self, x: Array, training: bool = False) -> Array:
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, grad: Array) -> Array:
        out = grad
        for layer in reversed(self.layers):
            out = layer.backward(out)
        return out

    def parameters(self) -> list[tuple[Array, Array]]:
        return [pg for layer in self.layers for pg in layer.parameters()]

    def get_state(self) -> dict[str, Array]:
        state: dict[str, Array] = {}
        d_idx = 0
        ln_idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                state[f"W{d_idx}"] = layer.W.copy()
                state[f"b{d_idx}"] = layer.b.copy()
                d_idx += 1
            elif isinstance(layer, LayerNorm):
                state[f"gamma{ln_idx}"] = layer.gamma.copy()
                state[f"beta{ln_idx}"] = layer.beta.copy()
                ln_idx += 1
        return state

    def set_state(self, state: dict[str, Array]) -> None:
        d_idx = 0
        ln_idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.W[...] = state[f"W{d_idx}"]
                layer.b[...] = state[f"b{d_idx}"]
                d_idx += 1
            elif isinstance(layer, LayerNorm):
                layer.gamma[...] = state[f"gamma{ln_idx}"]
                layer.beta[...] = state[f"beta{ln_idx}"]
                ln_idx += 1


@dataclass(frozen=True)
class MultimodalMatch:
    item: str
    score: float
    rank: int


class VisionLanguageModel:
    def __init__(
        self,
        image_size: tuple[int, int] = (16, 16),
        vocab_buckets: int = 512,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        temperature: float = 0.07,
        seed: int = 0,
    ) -> None:
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.vocab_buckets = int(vocab_buckets)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.temperature = float(temperature)
        self.meta: dict[str, Any] = {}

        rng = np.random.default_rng(seed)
        self.vision_backbone = VisionBackbone(self.image_size)
        self.text_backbone = TextBackbone(self.vocab_buckets)
        self.vision_tower = ProjectionTower(
            self.vision_backbone.out_dim, self.hidden_dim, self.embed_dim, rng
        )
        self.text_tower = ProjectionTower(
            self.text_backbone.out_dim, self.hidden_dim, self.embed_dim, rng
        )
        self._loss = InfoNCELoss(self.temperature)

    def encode_image(self, images: Any) -> Array:
        single = isinstance(images, (str, Path)) or (
            isinstance(images, np.ndarray)
            and (
                images.ndim == 1
                or (images.ndim == 2 and images.shape == self.image_size)
            )
        )
        batch = [images] if single else list(images)
        feats = self.vision_backbone.extract_batch(batch)
        proj = self.vision_tower.forward(feats, training=False)
        normed, _ = l2_normalize(proj)
        return normed[0] if single else normed

    def encode_text(self, texts: str | Sequence[str]) -> Array:
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        feats = self.text_backbone.extract_batch(batch)
        proj = self.text_tower.forward(feats, training=False)
        normed, _ = l2_normalize(proj)
        return normed[0] if single else normed

    def similarity(self, images: Any, texts: str | Sequence[str]) -> Array:
        img_emb = np.atleast_2d(self.encode_image(images))
        txt_emb = np.atleast_2d(self.encode_text(texts))
        return img_emb @ txt_emb.T

    def zero_shot_classify(
        self,
        image: Any,
        labels: Sequence[str],
        prompt_template: str = "a photo of a {}",
    ) -> list[tuple[str, float]]:
        if not labels:
            raise ValueError("labels must not be empty")
        prompts = [prompt_template.format(lbl) for lbl in labels]
        img_emb = np.atleast_2d(self.encode_image(image))
        txt_emb = np.atleast_2d(self.encode_text(prompts))
        logits = (img_emb @ txt_emb.T) / self.temperature
        probs = softmax(logits)[0]
        order = np.argsort(-probs)
        return [(labels[i], float(probs[i])) for i in order]

    def search_images(
        self,
        query: str,
        image_sources: Sequence[Any],
        top_k: int = 5,
    ) -> list[MultimodalMatch]:
        sims = self.similarity(image_sources, [query]).ravel()
        order = np.argsort(-sims)[:top_k]
        return [
            MultimodalMatch(str(image_sources[int(i)]), float(sims[int(i)]), rank + 1)
            for rank, i in enumerate(order)
        ]

    def search_texts(
        self,
        image: Any,
        texts: Sequence[str],
        top_k: int = 5,
    ) -> list[MultimodalMatch]:
        sims = self.similarity([image], texts).ravel()
        order = np.argsort(-sims)[:top_k]
        return [
            MultimodalMatch(texts[int(i)], float(sims[int(i)]), rank + 1)
            for rank, i in enumerate(order)
        ]

    def fit(
        self,
        images: Sequence[Any],
        texts: Sequence[str],
        *,
        labels: Sequence[Any] | None = None,
        epochs: int = 40,
        batch_size: int = 16,
        lr: float = 0.005,
        optimizer: str = "adamw",
        max_grad_norm: float = 1.0,
        seed: int = 0,
        verbose: int = 0,
    ) -> dict[str, list[float]]:
        if len(images) != len(texts) or len(images) == 0:
            raise ValueError("images and texts must have the same non-zero length")
        img_feats = self.vision_backbone.extract_batch(images)
        txt_feats = self.text_backbone.extract_batch(texts)
        n = len(img_feats)

        if labels is None:
            unique_texts = {t: i for i, t in enumerate(dict.fromkeys(texts))}
            lbl_arr = np.array([unique_texts[t] for t in texts])
        else:
            unique_lbls = {l: i for i, l in enumerate(dict.fromkeys(labels))}
            lbl_arr = np.array([unique_lbls[l] for l in labels])

        opt = make_optimizer(optimizer, lr) if optimizer != "adamw" else AdamW(lr=lr)
        params = self.vision_tower.parameters() + self.text_tower.parameters()
        rng = np.random.default_rng(seed)
        history: dict[str, list[float]] = {"loss": [], "alignment": []}

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            epoch_losses: list[float] = []
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) < 2 and n >= 2:
                    continue
                v_out = self.vision_tower.forward(img_feats[idx], training=True)
                t_out = self.text_tower.forward(txt_feats[idx], training=True)
                batch_loss = self._loss.forward(v_out, t_out, labels=lbl_arr[idx])
                dv, dt = self._loss.backward()
                self.vision_tower.backward(dv)
                self.text_tower.backward(dt)
                if max_grad_norm > 0:
                    clip_grad_norm(params, max_grad_norm)
                opt.step(params)
                epoch_losses.append(batch_loss)

            v_all, _ = l2_normalize(self.vision_tower.forward(img_feats, training=False))
            t_all, _ = l2_normalize(self.text_tower.forward(txt_feats, training=False))
            sim_matrix = v_all @ t_all.T
            top_matches = lbl_arr[sim_matrix.argmax(axis=1)] == lbl_arr
            align_acc = float(top_matches.mean())
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            history["loss"].append(mean_loss)
            history["alignment"].append(align_acc)

            if verbose and (epoch % verbose == 0 or epoch == epochs):
                print(
                    f"epoch {epoch:>4}/{epochs}  loss {mean_loss:.4f}  alignment {align_acc:.3f}"
                )
        return history

    def fit_folder(
        self,
        root: str | Path,
        *,
        epochs: int = 40,
        batch_size: int = 16,
        lr: float = 0.005,
        seed: int = 0,
        verbose: int = 0,
    ) -> dict[str, list[float]]:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"image folder not found: {root}")
        class_dirs = sorted(
            p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        if len(class_dirs) < 2:
            raise ValueError(f"{root} needs at least two class subfolders")

        templates = (
            "{}",
            "a photo of a {}",
            "{} shape",
            "an image showing a {}",
        )
        images: list[Path] = []
        texts: list[str] = []
        labels: list[int] = []
        class_names: list[str] = []

        for label_idx, folder in enumerate(class_dirs):
            name = folder.name.replace("_", " ")
            class_names.append(folder.name)
            files = [
                p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            for i, path in enumerate(files):
                images.append(path)
                texts.append(templates[i % len(templates)].format(name))
                labels.append(label_idx)

        self.meta["class_names"] = class_names
        return self.fit(
            images,
            texts,
            labels=labels,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            verbose=verbose,
        )

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> None:
        config = {
            "image_size": list(self.image_size),
            "vocab_buckets": self.vocab_buckets,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "temperature": self.temperature,
            "meta": meta if meta is not None else self.meta,
        }
        arrays: dict[str, Array] = {"vlm_config": np.array(json.dumps(config))}
        for k, v in self.vision_tower.get_state().items():
            arrays[f"vis_{k}"] = v
        for k, v in self.text_tower.get_state().items():
            arrays[f"txt_{k}"] = v
        np.savez_compressed(Path(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> VisionLanguageModel:
        with np.load(Path(path), allow_pickle=False) as data:
            if "vlm_config" not in data:
                raise ValueError(f"{path} is not a valid VisionLanguageModel file")
            cfg = json.loads(str(data["vlm_config"]))
            vlm = cls(
                image_size=tuple(cfg["image_size"]),
                vocab_buckets=cfg["vocab_buckets"],
                hidden_dim=cfg["hidden_dim"],
                embed_dim=cfg["embed_dim"],
                temperature=cfg["temperature"],
            )
            vlm.meta = cfg.get("meta", {})
            vis_state = {k[4:]: data[k] for k in data.files if k.startswith("vis_")}
            txt_state = {k[4:]: data[k] for k in data.files if k.startswith("txt_")}
            vlm.vision_tower.set_state(vis_state)
            vlm.text_tower.set_state(txt_state)
        return vlm

    def __repr__(self) -> str:
        h, w = self.image_size
        return (
            f"VisionLanguageModel(image={h}x{w}, hidden={self.hidden_dim}, "
            f"embed={self.embed_dim})"
        )

