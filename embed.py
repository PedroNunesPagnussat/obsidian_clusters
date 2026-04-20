"""Embed notes with sentence-transformers, with a (path, mtime, size) cache."""

from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_FENCED_CODE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1[ \t]*$", re.DOTALL | re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


@lru_cache(maxsize=4)
def load_model(name: str, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


def read_note(path: Path) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    # Strip fenced code blocks (``` ... ``` or ~~~ ... ~~~) of any language.
    text = _FENCED_CODE_RE.sub("", text)
    # Strip inline `code` spans.
    text = _INLINE_CODE_RE.sub("", text)
    # [[Target|Display]] -> Display ;  [[Target]] -> Target
    text = _WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    return text.strip()


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _atomic_write_npy(arr: np.ndarray, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(tmp, path)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def embed_notes(
    paths: Iterable[Path],
    model,
    cache_dir: Path,
    batch_size: int = 32,
    use_cache: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return (embeddings, dataframe) aligned row-wise. Uses cache when present."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "files.parquet"

    cached_emb: np.ndarray | None = None
    cached_meta: pd.DataFrame | None = None
    if use_cache and emb_path.exists() and meta_path.exists():
        try:
            cached_emb = np.load(emb_path)
            cached_meta = pd.read_parquet(meta_path)
            if len(cached_meta) != len(cached_emb):
                cached_emb, cached_meta = None, None
        except Exception:
            cached_emb, cached_meta = None, None

    cache_lookup: dict[tuple, int] = {}
    if cached_meta is not None:
        for i, row in enumerate(cached_meta.itertuples(index=False)):
            cache_lookup[(row.path, int(row.mtime), int(row.size))] = i

    rows = []
    embeddings: list[np.ndarray] = []
    to_embed_idx: list[int] = []
    to_embed_texts: list[str] = []

    paths = list(paths)
    for p in tqdm(paths, desc="scanning + cache lookup", unit="note"):
        p = Path(p)
        try:
            st = p.stat()
        except OSError:
            continue
        key = (str(p), int(st.st_mtime), int(st.st_size))
        row_idx = len(rows)
        rows.append({"path": key[0], "mtime": key[1], "size": key[2], "text_hash": ""})

        cached_i = cache_lookup.get(key) if cached_emb is not None else None
        if cached_i is not None:
            embeddings.append(cached_emb[cached_i])
            rows[-1]["text_hash"] = cached_meta.iloc[cached_i]["text_hash"]
        else:
            text = read_note(p)
            if not text:
                # placeholder — will be filled with zero vector after we know dim
                to_embed_idx.append(row_idx)
                to_embed_texts.append(" ")
                rows[-1]["text_hash"] = _text_hash("")
            else:
                to_embed_idx.append(row_idx)
                to_embed_texts.append(text)
                rows[-1]["text_hash"] = _text_hash(text)
            embeddings.append(None)  # type: ignore[arg-type]

    if to_embed_texts:
        print(f"embedding {len(to_embed_texts)} notes (cache hits: {len(rows) - len(to_embed_texts)})")
        new_vecs = model.encode(
            to_embed_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        for slot, vec in zip(to_embed_idx, new_vecs):
            embeddings[slot] = vec
    else:
        print(f"embedding 0 notes (cache hits: {len(rows)})")

    if not embeddings:
        raise ValueError("no notes to embed")

    matrix = np.vstack(embeddings).astype(np.float32)
    df = pd.DataFrame(rows)

    _atomic_write_npy(matrix, emb_path)
    _atomic_write_parquet(df, meta_path)

    return matrix, df
