"""CLI entry point: discover -> embed -> reduce (2D+3D) -> cluster -> plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cluster import centroid_indices, cluster, reduce_2d, reduce_3d
from discover import DEFAULT_EXCLUDES, iter_notes
from embed import embed_notes, load_model
from plot import (
    plot_clusters,
    plot_clusters_3d,
    plot_embeddings,
    plot_embeddings_3d,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed and cluster Obsidian notes locally.")
    p.add_argument("--notes-dir", type=Path, default=Path("~/notes"))
    p.add_argument("--exclude", nargs="*", default=sorted(DEFAULT_EXCLUDES))
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--out", type=Path, default=Path("charts"))
    p.add_argument("--cache-dir", type=Path, default=Path("embeddings"))
    p.add_argument("--max-bytes", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu", help="torch device: cpu, cuda, mps (default cpu)")
    p.add_argument("--no-3d", action="store_true", help="skip 3D charts")
    p.add_argument("--no-cache", action="store_true", help="ignore existing embedding cache and re-embed all notes")
    p.add_argument("--cluster-threshold", type=float, default=0.4,
                   help="cosine distance threshold for agglomerative clustering (smaller -> more clusters)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    notes_dir = args.notes_dir.expanduser()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"discovering notes in {notes_dir} (excluding {args.exclude})")
    paths = sorted(
        tqdm(
            iter_notes(notes_dir, excludes=args.exclude, max_bytes=args.max_bytes),
            desc="discovering",
            unit="file",
        )
    )
    if not paths:
        raise SystemExit(f"no notes found in {notes_dir}")
    print(f"found {len(paths)} notes")

    print(f"loading model {args.model} on {args.device}")
    model = load_model(args.model, device=args.device)

    emb, df = embed_notes(
        paths, model,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        use_cache=not args.no_cache,
    )

    print(f"clustering on full embeddings (agglomerative, cosine, threshold={args.cluster_threshold})")
    labels = cluster(emb, distance_threshold=args.cluster_threshold)

    print("running t-SNE 2D reduction")
    xy = reduce_2d(emb)

    print("finding centroid-nearest notes per cluster")
    centroids = centroid_indices(emb, labels)

    print("plotting 2D maps")
    plot_embeddings(xy, out_dir / "embeddings.png")
    plot_clusters(xy, labels, out_dir / "clusters.png", centroid_idx=centroids)

    if not args.no_3d:
        print("running t-SNE 3D reduction")
        xyz = reduce_3d(emb)
        print("plotting 3D maps")
        plot_embeddings_3d(xyz, out_dir / "embeddings_3d.png")
        plot_clusters_3d(xyz, labels, out_dir / "clusters_3d.png", centroid_idx=centroids)

    is_centroid = np.zeros(len(labels), dtype=bool)
    for i in centroids.values():
        is_centroid[i] = True
    out_df = pd.DataFrame({
        "cluster_id": labels,
        "is_centroid": is_centroid,
        "path": df["path"].values,
    }).sort_values(["cluster_id", "is_centroid", "path"], ascending=[True, False, True])
    csv_path = out_dir / "clusters.csv"
    out_df.to_csv(csv_path, index=False)

    centroids_csv = out_dir / "centroids.csv"
    pd.DataFrame(
        [{"cluster_id": lab, "path": df["path"].iloc[i]} for lab, i in centroids.items()]
    ).to_csv(centroids_csv, index=False)

    n = len(paths)
    n_clusters = len(centroids)
    n_noise = int((labels == -1).sum())
    print(
        f"\n{n} notes · {n_clusters} clusters · {n_noise} noise points"
        f"\nwrote: {out_dir / 'embeddings.png'}, {out_dir / 'clusters.png'}"
    )
    if not args.no_3d:
        print(f"       {out_dir / 'embeddings_3d.png'}, {out_dir / 'clusters_3d.png'}")
    print(f"       {csv_path}, {centroids_csv}")
    if centroids:
        print("\ncentroid notes per cluster:")
        for lab, i in centroids.items():
            print(f"  c{lab}: {df['path'].iloc[i]}")


if __name__ == "__main__":
    main()
