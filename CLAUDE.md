# CLAUDE.md

Guidance for Claude Code working in this repo.

## Purpose

Local pipeline that walks an Obsidian vault, embeds each note (whole-file)
with a `sentence-transformers` model, and runs **BERTopic** (UMAP 10D →
HDBSCAN → c-TF-IDF) to produce keyword-labeled topics plus 2D scatter PNGs
and CSVs.

Single-file CLI in `main.py`.

## Common commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# default: ~/notes, BAAI/bge-small-en-v1.5, CPU
python main.py

# point at a vault
python main.py --notes-dir ~/notes

# tune clustering granularity
python main.py --min-cluster-size 3       # smaller clusters
python main.py --umap-neighbors 30        # broader topics

# swap embedding model (invalidates cache automatically)
python main.py --model sentence-transformers/all-MiniLM-L6-v2
```

No tests, linter, or build step.

## Architecture

```
walk_notes  ->  read_note (strip frontmatter)  ->  embed_all (cache)
   ->  run_bertopic (UMAP 10D + HDBSCAN + c-TF-IDF)
   ->  project_2d (separate UMAP, viz only)
   ->  write_outputs (PNGs + CSVs)
```

Key invariants:

- **Clustering happens in 10D UMAP space, not in the 2D projection.** The 2D
  UMAP is for plotting only. Don't conflate them.
- **Cache invalidation**: keyed on `(path, mtime_ns, size)`. The pickle also
  stores `model_name`; swapping `--model` drops the cache and re-embeds all.
- **Frontmatter is stripped** before embedding — the `---\n...\n---` block
  pollutes c-TF-IDF keywords (`tags`, `aliases`, etc.) otherwise.
- **Whole-note embeddings**: one vector per file. No chunking.
- **BERTopic with custom embeddings**: pass `embedding_model=None` plus
  `embeddings=...` to `fit_transform`.
- **Short-corpus guard**: if `n < 2 * min_cluster_size`, skip clustering and
  write only `embeddings.png`.
- **Determinism**: both UMAPs get `random_state=args.seed`. HDBSCAN is
  deterministic given input.

## Outputs

- `charts/embeddings.png` — 2D UMAP scatter, all gray.
- `charts/clusters.png` — same scatter, colored by topic, legend = top
  topics with top-3 keywords.
- `charts/clusters.csv` — `path, topic_id, topic_keywords` (top-5).
- `charts/topics.csv` — `topic_id, count, name, top_keywords` (top-10).
- `embeddings/cache.pkl` — pickled `{model_name, vectors: {key: vec}}`.

## Gotchas

- BERTopic, UMAP, HDBSCAN are heavy installs; CPU is the default target.
- UMAP needs `n_neighbors < n_samples`; `run_bertopic` clamps this.
- Topic `-1` = noise (HDBSCAN); plotted in light gray, excluded from CSV
  keyword lookup.
