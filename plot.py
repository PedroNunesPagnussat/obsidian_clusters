"""Matplotlib renderers for 2D and 3D embedding maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

STYLE = "seaborn-v0_8-whitegrid"
BG = "#fafafa"
FG_SINGLE = "#3a7ca5"
NOISE_COLOR = "#cfcfcf"


def _palette(n: int) -> list:
    if n <= 10:
        return list(plt.get_cmap("tab10").colors)[:n]
    if n <= 20:
        return list(plt.get_cmap("tab20").colors)[:n]
    # many clusters: evenly sample hsv
    cmap = plt.get_cmap("hsv")
    return [cmap(i / n) for i in range(n)]


def _clean_2d(ax):
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(BG)


def _clean_3d(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("#dddddd")
        axis.pane.set_facecolor(BG)
    ax.grid(False)


def plot_embeddings(xy: np.ndarray, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor("white")
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=18, alpha=0.75, color=FG_SINGLE,
            edgecolors="white", linewidths=0.4,
        )
        ax.set_title(f"{len(xy)} notes — t-SNE 2D", fontsize=14, pad=14)
        _clean_2d(ax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, facecolor="white")
        plt.close(fig)


def plot_clusters(
    xy: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    centroid_idx: dict[int, int] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 9))
        fig.patch.set_facecolor("white")

        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(
                xy[noise_mask, 0], xy[noise_mask, 1],
                s=12, alpha=0.35, color=NOISE_COLOR, label="noise",
                edgecolors="none",
            )

        unique = sorted(set(labels.tolist()) - {-1})
        palette = _palette(len(unique))
        for color, lab in zip(palette, unique):
            m = labels == lab
            ax.scatter(
                xy[m, 0], xy[m, 1],
                s=22, alpha=0.85, color=color, label=f"c{lab} (n={int(m.sum())})",
                edgecolors="white", linewidths=0.4,
            )

        if centroid_idx:
            for lab, i in centroid_idx.items():
                ax.scatter(
                    xy[i, 0], xy[i, 1],
                    s=140, marker="*", color="black",
                    edgecolors="white", linewidths=1.0, zorder=5,
                )

        n_clusters = len(unique)
        n_noise = int(noise_mask.sum())
        ax.set_title(
            f"{n_clusters} clusters · {n_noise} noise / {len(xy)} notes",
            fontsize=14, pad=14,
        )
        _clean_2d(ax)
        if 0 < n_clusters <= 15:
            ax.legend(loc="best", fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, facecolor="white")
        plt.close(fig)


def plot_embeddings_3d(xyz: np.ndarray, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.style.context(STYLE):
        fig = plt.figure(figsize=(10, 9))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=14, alpha=0.75, color=FG_SINGLE,
            edgecolors="white", linewidths=0.3,
        )
        ax.set_title(f"{len(xyz)} notes — t-SNE 3D", fontsize=14, pad=14)
        _clean_3d(ax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, facecolor="white")
        plt.close(fig)


def plot_clusters_3d(
    xyz: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    centroid_idx: dict[int, int] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with plt.style.context(STYLE):
        fig = plt.figure(figsize=(10, 9))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection="3d")

        noise_mask = labels == -1
        if noise_mask.any():
            ax.scatter(
                xyz[noise_mask, 0], xyz[noise_mask, 1], xyz[noise_mask, 2],
                s=10, alpha=0.3, color=NOISE_COLOR, label="noise",
            )

        unique = sorted(set(labels.tolist()) - {-1})
        palette = _palette(len(unique))
        for color, lab in zip(palette, unique):
            m = labels == lab
            ax.scatter(
                xyz[m, 0], xyz[m, 1], xyz[m, 2],
                s=18, alpha=0.85, color=color, label=f"c{lab} (n={int(m.sum())})",
                edgecolors="white", linewidths=0.3,
            )

        if centroid_idx:
            for lab, i in centroid_idx.items():
                ax.scatter(
                    xyz[i, 0], xyz[i, 1], xyz[i, 2],
                    s=130, marker="*", color="black",
                    edgecolors="white", linewidths=1.0,
                )

        n_clusters = len(unique)
        n_noise = int(noise_mask.sum())
        ax.set_title(
            f"{n_clusters} clusters · {n_noise} noise / {len(xyz)} notes",
            fontsize=14, pad=14,
        )
        _clean_3d(ax)
        if 0 < n_clusters <= 15:
            ax.legend(loc="best", fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, facecolor="white")
        plt.close(fig)
