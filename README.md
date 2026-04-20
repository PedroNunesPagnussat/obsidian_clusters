# obsidian_clusters

Embed and cluster Obsidian notes using a local embedding model, then visualize.

## Pipeline

1. Walk a notes directory (default `~/notes`), excluding `.obsidian`, `.git`, `000_ARCHAIVE`.
2. Embed each note with `sentence-transformers` (`all-MiniLM-L6-v2` by default).
3. Reduce embeddings to 2D with UMAP.
4. Cluster automatically with HDBSCAN (no preset K).
5. Write two PNGs (`embeddings.png`, `clusters.png`) and a `clusters.csv`.

Embeddings are cached on `(path, mtime, size)`, so re-runs only embed changed/new files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --notes-dir ~/notes
```

Options:

```
--notes-dir DIR        notes root (default ~/notes)
--exclude NAME ...     directory names to skip (default: .obsidian .git 000_ARCHAIVE)
--model NAME           sentence-transformers model name (default all-MiniLM-L6-v2)
--out DIR              output dir for PNGs/CSV (default charts/)
--cache-dir DIR        embedding cache (default embeddings/)
--max-bytes N          skip files larger than this (default 1_000_000)
--batch-size N         embedding batch size (default 32)
```

Outputs:

- `charts/embeddings.png` — 2D map of all notes.
- `charts/clusters.png` — same map colored by HDBSCAN cluster.
- `charts/clusters.csv` — `path, cluster_id` for inspection.
