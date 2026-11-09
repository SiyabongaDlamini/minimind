from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

Array = np.ndarray

TEXT_EXTENSIONS = (".txt", ".md", ".rst", ".csv", ".json", ".log", ".tsv")
_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
        "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our",
        "she", "that", "the", "their", "them", "they", "this", "to", "was", "we", "were", "what",
        "when", "where", "which", "who", "why", "will", "with", "you", "your", "do", "does",
        "did", "can", "how", "about", "so", "than", "then", "there", "these", "those", "not",
        "no",
    ]
)
_WORD = re.compile(r"[a-z0-9]+")
_HEADING = re.compile(r"^(#{1,6}\s.*)$", re.MULTILINE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}|\n(?=[-*•\d])")


def read_document(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if not path.is_file():
        raise FileNotFoundError(f"document not found: {path}")
    if suffix in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        return _read_docx(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(
        f"unsupported file type {suffix!r} for {path.name}. Supported: "
        f"{', '.join(TEXT_EXTENSIONS + ('.docx', '.pdf'))}. For scanned images, run OCR first "
        "(for example `tesseract scan.png out`) and add the resulting text file."
    )


def _read_docx(path: Path) -> str:
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    with zipfile.ZipFile(path) as zf:
        root = ElementTree.fromstring(zf.read("word/document.xml"))
    paragraphs = []
    for para in root.iter(f"{ns}p"):
        text = "".join(node.text or "" for node in para.iter(f"{ns}t"))
        if text.strip():
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("reading PDFs needs pypdf: pip install 'minimind[docs]'") from exc
    pages = [(page.extract_text() or "") for page in PdfReader(str(path)).pages]
    text = "\n\n".join(pages).strip()
    if not text:
        raise ValueError(f"{path.name} contains no extractable text (is it a scanned image?)")
    return text


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in _WORD.findall(text.lower()):
        if word in _STOPWORDS:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                break
        tokens.append(word)
    return tokens


def split_sentences(text: str) -> list[str]:
    text = _HEADING.sub(r"\1\n\n", text)
    return [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]


def chunk_text(text: str, max_words: int = 90) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if current and count + words > max_words:
            chunks.append("\n\n".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        chunks.append("\n\n".join(current))
    return chunks


@dataclass(frozen=True)
class Passage:
    source: str
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    confidence: float
    passages: tuple[Passage, ...]
    found: bool

    def __str__(self) -> str:
        if not self.found:
            return self.text
        sources = ", ".join(dict.fromkeys(p.source for p in self.passages))
        return f"{self.text}\n\nSources: {sources}"


class KnowledgeBase:
    def __init__(self, max_words: int = 90, min_score: float = 0.12) -> None:
        self.max_words = max_words
        self.min_score = min_score
        self._chunks: list[tuple[str, str]] = []
        self._vocab: dict[str, int] = {}
        self._idf: Array = np.zeros(0)
        self._matrix: Array = np.zeros((0, 0))
        self._dirty = False

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(src for src, _ in self._chunks))

    def add_text(self, text: str, source: str = "text") -> int:
        chunks = chunk_text(text, self.max_words)
        self._chunks.extend((source, c) for c in chunks)
        self._dirty = self._dirty or bool(chunks)
        return len(chunks)

    def add_file(self, path: str | Path) -> int:
        path = Path(path)
        return self.add_text(read_document(path), source=path.name)

    def add_paths(self, paths: Iterable[str | Path]) -> dict[str, int]:
        report: dict[str, int] = {}
        for raw in paths:
            raw = Path(raw)
            files = sorted(p for p in raw.rglob("*") if p.is_file()) if raw.is_dir() else [raw]
            for f in files:
                try:
                    report[f.name] = self.add_file(f)
                except (ValueError, ImportError, zipfile.BadZipFile):
                    report[f.name] = -1
        return report

    def search(self, question: str, k: int = 3) -> list[Passage]:
        if not self._chunks:
            return []
        self._build()
        query = self._vectorize([question])[0]
        if not query.any():
            return []
        scores = self._matrix @ query
        order = np.argsort(-scores)[:k]
        return [
            Passage(self._chunks[i][0], self._chunks[i][1], float(scores[i]))
            for i in order
            if scores[i] > 0
        ]

    def ask(self, question: str, k: int = 3, max_sentences: int = 3) -> Answer:
        if not self._chunks:
            return Answer("No documents have been added yet.", 0.0, (), False)
        passages = self.search(question, k)
        if not passages or passages[0].score < self.min_score:
            return Answer(
                "I couldn't find anything about that in your documents.",
                passages[0].score if passages else 0.0,
                tuple(passages),
                False,
            )
        candidates = [(s, p) for p in passages for s in split_sentences(p.text)]
        query = self._vectorize([question])[0]
        sent_scores = self._vectorize([s for s, _ in candidates]) @ query
        chosen = [int(i) for i in np.argsort(-sent_scores)[:max_sentences] if sent_scores[i] > 0]
        text = " ".join(candidates[i][0] for i in sorted(chosen))
        used = tuple(dict.fromkeys(candidates[i][1] for i in chosen))
        return Answer(text, passages[0].score, used or tuple(passages), True)

    def save(self, path: str | Path) -> None:
        payload = {
            "max_words": self.max_words,
            "min_score": self.min_score,
            "chunks": self._chunks,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeBase:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        kb = cls(payload["max_words"], payload["min_score"])
        kb._chunks = [(s, t) for s, t in payload["chunks"]]
        kb._dirty = bool(kb._chunks)
        return kb

    def _build(self) -> None:
        if not self._dirty:
            return
        token_lists = [tokenize(text) for _, text in self._chunks]
        vocab = sorted({t for tokens in token_lists for t in tokens})
        self._vocab = {t: i for i, t in enumerate(vocab)}
        doc_freq = np.zeros(len(vocab))
        for tokens in token_lists:
            for t in set(tokens):
                doc_freq[self._vocab[t]] += 1
        n = len(token_lists)
        self._idf = np.log((1 + n) / (1 + doc_freq)) + 1.0
        self._matrix = self._vectorize_tokens(token_lists)
        self._dirty = False

    def _vectorize(self, texts: list[str]) -> Array:
        return self._vectorize_tokens([tokenize(t) for t in texts])

    def _vectorize_tokens(self, token_lists: list[list[str]]) -> Array:
        mat = np.zeros((len(token_lists), len(self._vocab)))
        for row, tokens in enumerate(token_lists):
            for term, count in Counter(tokens).items():
                col = self._vocab.get(term)
                if col is not None:
                    mat[row, col] = (1.0 + math.log(count)) * self._idf[col]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(norms == 0, 1.0, norms)
