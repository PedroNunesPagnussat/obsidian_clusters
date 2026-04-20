# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Local pipeline that walks an Obsidian vault, embeds each note with a local `sentence-transformers` model, clusters the embeddings, and renders 2D/3D scatter plots plus CSVs.

## Common commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run against default ~/notes
python main.py

# force CPU (default) vs GPU (only if torch build supports the card)
python main.py --device cpu
python main.py --device cuda

# bypass embedding cache (re-embed everything)
python main.py --no-cache

# skip 3D charts (faster; t-SNE 3D is the slow step)
python main.py --no-3d

# tune clustering granularity (smaller threshold -> more, tighter clusters)
python main.py --cluster-threshold 0.35
```

There is no test suite, linter config, or build step — it's a small CLI.

## Architecture

Single-process, linear pipeline wired in `main.py`:

```
discover.iter_notes  ->  embed.embed_notes  ->  cluster.cluster      ->  plot.plot_clusters(_3d)
 (os.walk, prune by         (cache keyed on       (Agglomerative,          (matplotlib; centroids
  excluded dir names,        path+mtime+size;      cosine, auto-K            marked with black stars)
  size-filtered)             writes .npy+parquet)  via distance_threshold
                                                   on FULL embeddings)
                                                          |
                                                          v
                                              cluster.reduce_2d / reduce_3d
                                              (t-SNE, cosine, PCA init —
                                               visualization only)
```

Key design points that aren't obvious from any single file:

- **Clustering runs on the full 384-dim embeddings, not on the t-SNE coords.** t-SNE is strictly for visualization. If you change the clustering algorithm, keep this separation — clustering in 2D t-SNE space would distort results.
- **Centroids are computed in embedding space** (`cluster.centroid_indices`): mean of unit-normalized vectors in a cluster, then argmax cosine similarity. This is independent of the reducer, so swapping t-SNE for something else doesn't affect which note is labeled the centroid.
- **Cache is keyed on `(path, mtime, size)`** (`embed.embed_notes`). `read_note` strips fenced code blocks, inline code, and Obsidian wikilinks (keeping display text) before embedding. **Frontmatter is intentionally kept.** The cache stores a `text_hash` column but does not currently key on it — mtime/size is the invalidation signal.
- **Atomic cache writes**: `np.save` appends `.npy` automatically, which breaks naive tmp-rename. `_atomic_write_npy` works around this by writing through an open file handle. Don't revert to `np.save(tmp_path, ...)`.
- **Directory excludes apply at any depth**, not just the top level — `os.walk`'s `dirnames` list is pruned in place on every iteration. Default excludes: `.obsidian`, `.git`, `000_META`.
- **Model loading is `lru_cache`'d** by `(name, device)` in `embed.load_model`, so repeated calls in one process are free.

## Output layout

- `charts/embeddings.png`, `charts/clusters.png` — 2D t-SNE scatter.
- `charts/embeddings_3d.png`, `charts/clusters_3d.png` — 3D t-SNE scatter (unless `--no-3d`).
- `charts/clusters.csv` — `cluster_id, is_centroid, path`, sorted by cluster then centroid-first.
- `charts/centroids.csv` — one row per cluster with the centroid-nearest path.
- `embeddings/embeddings.npy` + `embeddings/files.parquet` — cache.

## Gotchas

- GPU support depends on the PyTorch build matching the card's compute capability. A GTX 1060 (sm_61) will fail with `CUDA error: no kernel image is available` on recent PyTorch wheels — default `--device cpu` avoids this.
- Agglomerative clustering with `metric="cosine"` requires `linkage="average"` (or `complete`/`single`) — not `ward`.
- t-SNE's `perplexity` must be less than `n_samples`; `cluster._tsne` clamps it to `min(30, (n-1)/3)`.
