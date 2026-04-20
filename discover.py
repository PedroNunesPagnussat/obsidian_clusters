"""Walk a notes directory, yielding paths to candidate note files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator

DEFAULT_EXCLUDES = {".obsidian", ".git", "000_META", ".claude", ".trash"}
DEFAULT_EXTENSIONS = (".md",)
DEFAULT_MAX_BYTES = 1_000_000  # 1 MB


def iter_notes(
    root: Path,
    excludes: Iterable[str] = DEFAULT_EXCLUDES,
    extensions: Iterable[str] = DEFAULT_EXTENSIONS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Iterator[Path]:
    """Yield note files under `root`, pruning excluded directory names anywhere in the tree.

    Skips empty files and files larger than `max_bytes`.
    """
    root = Path(root).expanduser().resolve()
    excludes = set(excludes)
    extensions = tuple(e.lower() for e in extensions)

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded dirs in place so os.walk does not descend into them.
        dirnames[:] = [d for d in dirnames if d not in excludes]

        for fname in filenames:
            if not fname.lower().endswith(extensions):
                continue
            fpath = Path(dirpath) / fname
            try:
                size = fpath.stat().st_size
            except OSError:
                continue
            if size == 0 or size > max_bytes:
                continue
            yield fpath
