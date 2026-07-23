"""検索の埋め込み空間を2Dに落として1問ぶん可視化する（診断用）。

灰 = コーパス全チャンク / 緑 = 正解記事のチャンク / 赤縁 = top-kヒット / 星 = クエリ。
AND質問で「2記事目のチャンクがクエリ近傍のどこにいるか」を見るのが主目的
（docs/retrieval-experiment-plan.md §2 診断1 の図）。

使い方:
  uv run python experiments/viz_retrieval.py --id forum_76963
  uv run python experiments/viz_retrieval.py --id old_04 --collection music_theory
出力: data/eval/viz/<id>_<collection>.png
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from music_rag import config, embedder, retriever


def load_question(qid: str) -> dict:
    rows = json.loads((config.EVAL_DIR / "eval_set_merged.json").read_text(encoding="utf-8"))
    for r in rows:
        if r["id"] == qid:
            return r
    raise SystemExit(f"id={qid} が eval_set_merged.json に無い")


def scroll_all(collection: str) -> tuple[np.ndarray, list[dict]]:
    """collection の全ポイントをベクトル付きで取得する。"""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    vecs, payloads, offset = [], [], None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            vecs.append(p.vector)
            payloads.append(p.payload)
        if offset is None:
            break
    return np.asarray(vecs, dtype=np.float32), payloads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", required=True, help="eval set の質問id（例: forum_76963）")
    ap.add_argument("--collection", default=config.COLLECTION_NAME)
    ap.add_argument("--k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    row = load_question(args.id)
    expected = set(row["expected_source"])
    print(f"Q: {row['question'][:60]}...")
    print(f"expected ({row['match_type']}): {sorted(expected)}")

    qvec = np.asarray(embedder.embed_query(row["question"]), dtype=np.float32)
    hits = retriever.search(qvec.tolist(), top_k=args.k, collection=args.collection)
    hit_keys = {(h["source"], h["chunk_index"]) for h in hits}
    print("hits:", [(h["source"].split("_")[-1], round(h["score"], 3)) for h in hits])

    vecs, payloads = scroll_all(args.collection)
    print(f"corpus: {len(vecs)} points")

    # t-SNE はクエリも含めて一緒に埋め込む（transform が無いため）。
    # 高次元のまま t-SNE に食わせると遅いので PCA で50次元に前処理（定石）。
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = np.vstack([vecs, qvec[None, :]])
    X50 = PCA(n_components=50, random_state=0).fit_transform(X)
    XY = TSNE(n_components=2, random_state=0, perplexity=30, init="pca").fit_transform(X50)
    xy, q_xy = XY[:-1], XY[-1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Hiragino Sans"  # 日本語タイトルの豆腐化防止(macOS)

    is_expected = np.array([p.get("source") in expected for p in payloads])
    is_hit = np.array(
        [(p.get("source"), p.get("chunk_index")) in hit_keys for p in payloads]
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xy[:, 0], xy[:, 1], s=6, c="#cccccc", linewidths=0, label="全チャンク")
    ax.scatter(xy[is_expected, 0], xy[is_expected, 1], s=28, c="#2e8b57",
               linewidths=0, label="正解記事のチャンク")
    ax.scatter(xy[is_hit, 0], xy[is_hit, 1], s=90, facecolors="none",
               edgecolors="#d62728", linewidths=1.8, label=f"top-{args.k} ヒット")
    ax.scatter(q_xy[0], q_xy[1], s=260, marker="*", c="#111111", zorder=5, label="クエリ")

    # 正解記事ごとに重心へ記事名（slug末尾）を添える
    for src in sorted(expected):
        mask = np.array([p.get("source") == src for p in payloads])
        if mask.any():
            cx, cy = xy[mask, 0].mean(), xy[mask, 1].mean()
            ax.annotate(src.split("_")[-1], (cx, cy), fontsize=9,
                        color="#1a5e38", weight="bold")

    ax.set_title(f"[{row['id']}] {row['question'][:48]}…\n"
                 f"match_type={row['match_type']}  collection={args.collection}",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xticks([]), ax.set_yticks([])

    out_dir = config.EVAL_DIR / "viz"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{row['id']}_{args.collection}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
