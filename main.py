"""Embed Obsidian notes and cluster them with BERTopic.

Walks a vault, cleans Obsidian/Markdown boilerplate out of each note,
embeds the remaining prose with a multilingual sentence-transformers
model, runs BERTopic (UMAP 10D -> HDBSCAN -> c-TF-IDF + KeyBERTInspired
+ MMR) for keyword-labeled topics, and writes 2D scatter PNGs and
per-note / per-topic CSVs.

Clustering happens in 10D UMAP space; the 2D UMAP is used only for the plot.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_EXCLUDES = (".obsidian", ".git", ".trash", "node_modules", "000_META")

# Bump this any time CLEANERS changes — invalidates embedding cache.
CLEAN_VERSION = 1


# ---------- text cleaning ----------

FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)

# Order matters: code first (so wiki-links inside code aren't processed),
# HTML after code, wiki-links after embeds.
def _wikilink_sub(m: re.Match) -> str:
    target, alias = m.group(1), m.group(2)
    if alias:
        return alias
    return target.split("/")[-1].split("#")[-1]

CLEANERS: list[tuple[re.Pattern, object]] = [
    # 1. fenced code blocks (covers ```dataview, ```js, ```html, etc.)
    (re.compile(r"```.*?```", re.DOTALL), " "),
    # 2. Bases ~~~ blocks
    (re.compile(r"~~~.*?~~~", re.DOTALL), " "),
    # 3. inline dataview expressions  `=this.file.foo`
    (re.compile(r"`=[^`]*`"), " "),
    # 4. HTML tags
    (re.compile(r"<[^>]+>"), " "),
    # 5. HTML entities -> space
    (re.compile(r"&[a-zA-Z]+;|&#\d+;"), " "),
    # 6. inline code
    (re.compile(r"`[^`]+`"), " "),
    # 7. embeds: ![[...]] -> drop
    (re.compile(r"!\[\[[^\]]+\]\]"), " "),
    # 8. wikilinks: [[target|alias]] or [[target]] -> alias or target stem
    (re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]"), _wikilink_sub),
    # 9. markdown links: [text](url) -> text
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    # 10. tags: #foo -> foo (keep the word as content)
    (re.compile(r"(?<!\S)#([\w/-]+)"), r"\1"),
    # 11. callout headers: > [!note]+
    (re.compile(r"^\s*>\s*\[![^\]]+\][+-]?\s*", re.MULTILINE), " "),
    # 12. table separator rows
    (re.compile(r"^\s*\|?[\s|:-]+\|?\s*$", re.MULTILINE), " "),
    # 13. task checkboxes -> bullet
    (re.compile(r"^\s*[-*]\s*\[[ xX]\]\s*", re.MULTILINE), "- "),
    # 14. collapse whitespace
    (re.compile(r"\s+"), " "),
]


def clean_note(text: str) -> str:
    text = FRONTMATTER_RE.sub("", text, count=1)
    for pattern, repl in CLEANERS:
        text = pattern.sub(repl, text)
    return text.strip()


def read_note(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return clean_note(text)


# ---------- stopwords ----------

# Hardcoded Portuguese stopwords (~150). Avoids pulling in NLTK.
PT_STOPWORDS = {
    "a", "agora", "ainda", "alem", "algo", "alguem", "algum", "alguma", "algumas", "alguns",
    "ali", "ano", "antes", "ao", "aos", "apenas", "apos", "aquela", "aquelas", "aquele",
    "aqueles", "aqui", "aquilo", "as", "assim", "ate", "atraves", "bem", "cada", "cerca",
    "coisa", "com", "como", "contra", "contudo", "da", "dado", "dai", "dao", "daquela",
    "daquele", "dar", "das", "de", "dela", "delas", "dele", "deles", "demais", "depois",
    "des", "desde", "dessa", "dessas", "desse", "desses", "desta", "destas", "deste",
    "destes", "deve", "devem", "devera", "deverao", "deveria", "deveriam", "devo", "dia",
    "diante", "disse", "disso", "disto", "dito", "diz", "dizem", "do", "dois", "dos",
    "duas", "e", "ela", "elas", "ele", "eles", "em", "embora", "entao", "entre", "era",
    "eram", "essa", "essas", "esse", "esses", "esta", "estamos", "estao", "estar", "estas",
    "estava", "estavam", "este", "esteja", "estejam", "estes", "esteve", "estive",
    "estivemos", "estiver", "estiveram", "estivesse", "estou", "eu", "fazer", "fez",
    "fim", "foi", "fomos", "for", "fora", "foram", "forma", "fosse", "fossem", "fui",
    "ha", "haja", "hao", "havia", "isso", "isto", "ja", "la", "lhe", "lhes", "logo",
    "mais", "mas", "me", "mediante", "menos", "mesma", "mesmas", "mesmo", "mesmos", "meu",
    "meus", "minha", "minhas", "muita", "muitas", "muito", "muitos", "na", "nada", "nao",
    "nas", "nem", "nenhum", "nessa", "nesse", "nesta", "neste", "ninguem", "no", "nos",
    "nossa", "nossas", "nosso", "nossos", "num", "numa", "nunca", "o", "os", "ou", "outra",
    "outras", "outro", "outros", "para", "pela", "pelas", "pelo", "pelos", "perante",
    "pode", "podem", "podia", "pois", "por", "porem", "porque", "posso", "pouca", "poucas",
    "pouco", "poucos", "primeira", "primeiro", "qual", "quais", "quando", "quanta",
    "quantas", "quanto", "quantos", "que", "quem", "se", "seja", "sejam", "sem", "sempre",
    "sendo", "sera", "serao", "seria", "seriam", "seu", "seus", "si", "sido", "so",
    "sob", "sobre", "sou", "sua", "suas", "tal", "tambem", "tao", "te", "tem", "temos",
    "tendo", "tenha", "tenham", "tenho", "ter", "teu", "teus", "ti", "tido", "tinha",
    "tinham", "toda", "todas", "todo", "todos", "tu", "tua", "tuas", "tudo", "ultima",
    "ultimo", "um", "uma", "umas", "uns", "vai", "vao", "vem", "ver", "vez", "vos",
    "voce", "voces",
}


def build_stopwords() -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return sorted(set(ENGLISH_STOP_WORDS) | PT_STOPWORDS)


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed and cluster Obsidian notes with BERTopic.")
    p.add_argument("--notes-dir", type=Path, default=Path("~/notes"))
    p.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDES))
    p.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                   help="sentence-transformers model name. Default is multilingual; "
                        "for English-only vaults try BAAI/bge-small-en-v1.5.")
    p.add_argument("--out", type=Path, default=Path("charts"))
    p.add_argument("--cache-dir", type=Path, default=Path("embeddings"))
    p.add_argument("--max-bytes", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu", help="torch device for SentenceTransformer (cpu/cuda/mps)")
    p.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size")
    p.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors (clustering and 2D)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--legend-top", type=int, default=12, help="how many topics to show in the legend")
    return p.parse_args()


# ---------- discovery ----------

def walk_notes(notes_dir: Path, excludes: list[str], max_bytes: int) -> list[Path]:
    excludes_set = set(excludes)
    out: list[Path] = []
    for root, dirnames, filenames in os.walk(notes_dir):
        dirnames[:] = [d for d in dirnames if d not in excludes_set]
        for fn in filenames:
            if not fn.endswith(".md"):
                continue
            p = Path(root) / fn
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            out.append(p)
    return sorted(out)


# ---------- cache ----------

def cache_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size)


def load_cache(cache_dir: Path, model_name: str) -> dict[tuple, np.ndarray]:
    f = cache_dir / "cache.pkl"
    if not f.exists():
        return {}
    try:
        with f.open("rb") as fh:
            blob = pickle.load(fh)
    except (OSError, pickle.UnpicklingError, EOFError):
        return {}
    if not isinstance(blob, dict):
        return {}
    if blob.get("model_name") != model_name:
        return {}
    if blob.get("clean_version") != CLEAN_VERSION:
        return {}
    vectors = blob.get("vectors", {})
    return vectors if isinstance(vectors, dict) else {}


def save_cache(cache_dir: Path, model_name: str, vectors: dict[tuple, np.ndarray]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / "cache.pkl"
    tmp = f.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(
            {"model_name": model_name, "clean_version": CLEAN_VERSION, "vectors": vectors},
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(tmp, f)


# ---------- embedding ----------

def make_st_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def embed_all(
    paths: list[Path],
    texts: list[str],
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    st_model,
) -> np.ndarray:
    cache = load_cache(cache_dir, model_name)
    keys = [cache_key(p) for p in paths]

    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    print(f"cache: {len(keys) - len(missing_idx)} hit, {len(missing_idx)} to embed")

    if missing_idx:
        miss_texts = [texts[i] for i in missing_idx]
        new_vecs = st_model.encode(
            miss_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        for j, i in enumerate(missing_idx):
            cache[keys[i]] = new_vecs[j]
        save_cache(cache_dir, model_name, cache)

    return np.stack([cache[k] for k in keys]).astype(np.float32)


# ---------- BERTopic ----------

def run_bertopic(
    docs: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int,
    n_neighbors: int,
    seed: int,
    st_model,
):
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    n = len(docs)
    nn = max(2, min(n_neighbors, n - 1))

    umap_model = UMAP(
        n_neighbors=nn,
        n_components=min(10, max(2, n - 2)),
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=build_stopwords(),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.5,
        # ≥3 letters, no digits, allow accented Latin
        token_pattern=r"(?u)\b[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ-]{2,}\b",
    )
    representation_model = [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.3),
    ]
    topic_model = BERTopic(
        embedding_model=st_model,
        umap_model=umap_model,
        hdbscan_model=hdb,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    return topic_model, np.asarray(topics)


def project_2d(embeddings: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    from umap import UMAP

    n = len(embeddings)
    nn = max(2, min(n_neighbors, n - 1))
    reducer = UMAP(
        n_neighbors=nn,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


# ---------- output ----------

def topic_keywords(topic_model, topic_id: int, k: int = 5) -> list[str]:
    if topic_id == -1:
        return []
    words = topic_model.get_topic(topic_id) or []
    return [w for w, _ in words[:k]]


def topic_example_paths(
    paths: list[Path], topics: np.ndarray, embeddings: np.ndarray
) -> dict[int, str]:
    """For each topic, pick the note whose embedding is closest to the topic centroid."""
    out: dict[int, str] = {}
    for tid in sorted(set(int(t) for t in topics)):
        if tid == -1:
            continue
        idx = np.where(topics == tid)[0]
        if idx.size == 0:
            continue
        cluster_emb = embeddings[idx]
        centroid = cluster_emb.mean(axis=0)
        # cosine similarity (vectors are normalized, but normalize centroid)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        sims = cluster_emb @ centroid
        best = idx[int(np.argmax(sims))]
        out[tid] = paths[best].stem
    return out


def write_outputs(
    paths: list[Path],
    topics: np.ndarray,
    embeddings: np.ndarray,
    coords_2d: np.ndarray,
    topic_model,
    out_dir: Path,
    legend_top: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # embeddings.png — gray scatter only
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=8, c="0.5", alpha=0.5, linewidths=0)
    ax.set_title(f"Note embeddings (UMAP 2D), n={len(paths)}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "embeddings.png", dpi=150)
    plt.close(fig)

    info = topic_model.get_topic_info()
    info_sorted = info[info["Topic"] != -1].sort_values("Count", ascending=False)
    legend_topics = list(info_sorted["Topic"].head(legend_top))
    examples = topic_example_paths(paths, topics, embeddings)

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 8))
    noise_mask = topics == -1
    if noise_mask.any():
        ax.scatter(coords_2d[noise_mask, 0], coords_2d[noise_mask, 1],
                   s=6, c="0.75", alpha=0.35, linewidths=0, label="_noise")
    unique = sorted(set(int(t) for t in topics) - {-1})
    for idx, t in enumerate(unique):
        m = topics == t
        kw = topic_keywords(topic_model, t, k=3)
        ex = examples.get(t)
        label = None
        if t in legend_topics:
            tag = ", ".join(kw) if kw else "(no keywords)"
            label = f"{t} [{ex}]: {tag}" if ex else f"{t}: {tag}"
        ax.scatter(coords_2d[m, 0], coords_2d[m, 1],
                   s=12, color=cmap(idx % 20), alpha=0.85, linewidths=0,
                   label=label)
    ax.set_title(f"Topics (BERTopic, {len(unique)} topics, {int(noise_mask.sum())} noise)")
    ax.set_xticks([]); ax.set_yticks([])
    if legend_topics:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "clusters.png", dpi=150)
    plt.close(fig)

    # clusters.csv — per-note
    kw_by_topic = {t: ", ".join(topic_keywords(topic_model, t, k=5)) for t in unique}
    kw_by_topic[-1] = ""
    rows = [{"path": str(p), "topic_id": int(t), "topic_keywords": kw_by_topic[int(t)]}
            for p, t in zip(paths, topics)]
    pd.DataFrame(rows).sort_values(["topic_id", "path"]).to_csv(out_dir / "clusters.csv", index=False)

    # topics.csv — one row per topic, with a representative-note column
    topic_rows = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        topic_rows.append({
            "topic_id": tid,
            "count": int(row["Count"]),
            "name": row["Name"],
            "example_note": examples.get(tid, ""),
            "top_keywords": ", ".join(topic_keywords(topic_model, tid, k=10)),
        })
    pd.DataFrame(topic_rows).to_csv(out_dir / "topics.csv", index=False)


# ---------- main ----------

def main() -> None:
    args = parse_args()
    notes_dir = args.notes_dir.expanduser()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"discovering notes in {notes_dir} (excluding {args.exclude})")
    paths = walk_notes(notes_dir, args.exclude, args.max_bytes)
    if not paths:
        raise SystemExit(f"no .md notes found in {notes_dir}")
    print(f"found {len(paths)} notes")

    print("reading + cleaning notes")
    texts = [read_note(p) for p in tqdm(paths, unit="file")]

    print(f"loading model {args.model} on {args.device}")
    st_model = make_st_model(args.model, args.device)

    embeddings = embed_all(
        paths, texts,
        model_name=args.model,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        st_model=st_model,
    )

    if len(paths) < 2 * args.min_cluster_size:
        print(
            f"only {len(paths)} notes (< 2*min_cluster_size={2*args.min_cluster_size}); "
            "skipping clustering and writing embeddings.png only.",
            file=sys.stderr,
        )
        coords = project_2d(embeddings, args.umap_neighbors, args.seed)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(coords[:, 0], coords[:, 1], s=8, c="0.5", alpha=0.5, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_dir / "embeddings.png", dpi=150)
        plt.close(fig)
        return

    print(f"running BERTopic (min_cluster_size={args.min_cluster_size}, n_neighbors={args.umap_neighbors})")
    topic_model, topics = run_bertopic(
        texts, embeddings,
        min_cluster_size=args.min_cluster_size,
        n_neighbors=args.umap_neighbors,
        seed=args.seed,
        st_model=st_model,
    )

    print("projecting to 2D for visualization")
    coords_2d = project_2d(embeddings, args.umap_neighbors, args.seed)

    print("writing outputs")
    write_outputs(paths, topics, embeddings, coords_2d, topic_model, out_dir, args.legend_top)

    n_topics = int(((np.unique(topics)) != -1).sum())
    n_noise = int((topics == -1).sum())
    print(
        f"\n{len(paths)} notes · {n_topics} topics · {n_noise} noise"
        f"\nwrote: {out_dir/'embeddings.png'}, {out_dir/'clusters.png'},"
        f" {out_dir/'clusters.csv'}, {out_dir/'topics.csv'}"
    )


if __name__ == "__main__":
    main()
