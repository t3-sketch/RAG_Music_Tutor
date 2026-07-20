# %% [markdown]
# # experiments/explore.py — 評価結果を対話的に掘るスクラッチ（`# %%` セル形式）
#
# `.ipynb` ではなく普通の `.py`。VSCode の Jupyter 拡張が `# %%` をセル区切りとして認識し、
# 各セル上に「Run Cell」を出す。押すと裏で ipykernel が立ち上がり、そのセルだけを
# Interactive Window で実行する（変数は保持され、グラフはインライン表示）。
#
# `.ipynb` に対する利点: git diff が綺麗（コードだけ、出力/メタデータが乗らない）・
# 他スクリプトから `from experiments.explore import ...` できる。
#
# 前提:
# - dev 依存に ipykernel / pandas / umap-learn / matplotlib / scikit-learn（`uv add --dev` 済み）
# - **repo ルートから実行する**（config.DATA_DIR が cwd 基準の ./data のため）
# - Qdrant が起動していること（埋め込み可視化セルで scroll する）

# %%
# ── セットアップ ──
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from music_rag import config, embedder, retriever

plt.rcParams["font.family"] = "Hiragino Sans"  # 日本語タイトルの豆腐化防止(macOS)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 160)

EVAL = config.EVAL_DIR
EVAL_SET = json.loads((EVAL / "eval_set_merged.json").read_text(encoding="utf-8"))
BY_ID = {r["id"]: r for r in EVAL_SET}
print(f"eval set: {len(EVAL_SET)} 問  ({EVAL/'eval_set_merged.json'})")


# %% [markdown]
# ## 1. 検索スコア（scores_*.json）を DataFrame で眺める
# `uv run python experiments/evaluation.py` が吐く per-question を読む。
# ここにあるのは retrieval 層の指標（recall@k / strict_hit / MRR）で、**RAGAS の管轄外**。

# %%
SCORES_PATH = sorted(EVAL.glob("scores_2*.json"))[-1]  # 最新の日付を採用
scores = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
print(f"loaded {SCORES_PATH.name}  strategies={list(scores)}")


def scores_df(strategy: str) -> pd.DataFrame:
    return pd.DataFrame(scores[strategy]["per_question"])


df = scores_df("structure")
df[["id", "match_type", "source", "difficulty", "recall", "strict_hit", "reciprocal_rank"]]


# %% [markdown]
# ### 層別集計 — 全体平均は AND質問（多ソース比較）の弱点を隠すので match_type 別に割る

# %%
df.groupby("match_type")[["recall", "strict_hit", "reciprocal_rank"]].agg(["mean", "count"])


# %% [markdown]
# ### fixed vs structure を 1 問ずつ突き合わせ（paired diff の素）
# 差 `d_recall` が負の行 = structure で悪化した問題。どの match_type で動いたかを見る。

# %%
f = scores_df("fixed").set_index("id")
s = scores_df("structure").set_index("id")
cmp = pd.DataFrame({
    "match_type": s["match_type"],
    "recall_fixed": f["recall"],
    "recall_struct": s["recall"],
    "d_recall": (s["recall"] - f["recall"]).round(4),
    "d_mrr": (s["reciprocal_rank"] - f["reciprocal_rank"]).round(4),
})
cmp.sort_values("d_recall")  # 悪化が上、改善が下


# %% [markdown]
# ## 2. 「拾ってるのに捨てている」Qdrant の生 cos 類似度を実際に見る
# `evaluation.retrieve_with_text` は `{"text","source","score"}` を返すが、
# `evaluate_generation` は text だけ使い `score` を捨てている。ここでは score を残して並べる。
# これは BGE-M3 × Qdrant の cos で、RAGAS が内部で使う Gemini embedding の cos とは別物。
#
# `QID` を変えれば任意の質問を掘れる（AND質問の例を初期値に）。

# %%
QID = "forum_76963"  # ← 見たい質問idに変える
row = BY_ID[QID]
expected = set(row["expected_source"])
print(f"Q: {row['question']}")
print(f"expected ({row['match_type']}): {sorted(expected)}")

qvec = embedder.embed_query(row["question"])
hits = retriever.search(qvec, top_k=config.TOP_K, collection=config.COLLECTION_NAME)
hits_df = pd.DataFrame([{
    "rank": i + 1,
    "source_tail": h["source"].split("_")[-1],
    "cos_score": round(h["score"], 4),
    "is_correct": h["source"] in expected,
    "chunk_index": h["chunk_index"],
} for i, h in enumerate(hits)])
hits_df


# %% [markdown]
# ## 3. RAGAS（生成層）の 5 指標を 1 問ずつ見る
# `run_ragas` が吐いた json。faithfulness / answer_relevancy / answer_correctness /
# context_precision / context_recall。まだ 66 問版が無ければ過去の structure 版を読む。

# %%
metric_cols = [
    "faithfulness", "answer_relevancy", "answer_correctness",
    "context_precision", "context_recall",
]
ragas_files = sorted(EVAL.glob("ragas_*structure*.json")) or sorted(EVAL.glob("ragas_*.json"))
if ragas_files:
    RAGAS_PATH = ragas_files[-1]
    ragas = json.loads(RAGAS_PATH.read_text(encoding="utf-8"))
    print(f"loaded {RAGAS_PATH.name}  n={ragas['n']}")
    rdf = pd.DataFrame(ragas["per_question"])
    display_cols = ["question", *[c for c in metric_cols if c in rdf.columns]]
    rdf_view = rdf[display_cols]
else:
    print("ragas_*.json が無い。run_ragas 実行後にこのセルを回す")
    rdf, rdf_view = pd.DataFrame(), pd.DataFrame()
rdf_view


# %% [markdown]
# ### RAGAS 5 指標の分布（箱ひげ）— 平均だけでなくばらつきを見る

# %%
if not rdf.empty:
    present = [c for c in metric_cols if c in rdf.columns]
    rdf[present].plot(kind="box", figsize=(9, 4), rot=20,
                      title=f"RAGAS 5指標の分布 ({RAGAS_PATH.name})")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## 4. 埋め込み空間を UMAP で 2D に落として 1 問を可視化
#
# 灰 = 全チャンク / 緑 = 正解記事のチャンク / 赤縁 = top-k ヒット / 星 = クエリ。
# → まさに「リトリーブした top-K を色で / 正解ラベルを別色で」。
#
# UMAP は t-SNE と違い `transform` を持つ。**コーパスに一度 fit すれば、後から来た
# クエリ点を同じ 2D 空間へ写せる**（クエリ拡張の前後比較＝次セルに必須）。
# `scroll_all` は Qdrant 全ポイントをベクトル付きで取る（viz_retrieval.py と同じ発想）。

# %%
import umap  # noqa: E402


def scroll_all(collection: str):
    from qdrant_client import QdrantClient
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    vecs, payloads, offset = [], [], None
    while True:
        pts, offset = client.scroll(
            collection_name=collection, limit=512, offset=offset,
            with_payload=True, with_vectors=True,
        )
        for p in pts:
            vecs.append(p.vector)
            payloads.append(p.payload)
        if offset is None:
            break
    return np.asarray(vecs, dtype=np.float32), payloads


COLLECTION = config.COLLECTION_NAME
corpus_vecs, payloads = scroll_all(COLLECTION)
print(f"corpus: {len(corpus_vecs)} points @ {COLLECTION}")

# cos 空間で作った BGE-M3 ベクトルなので UMAP も metric="cosine"。
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0)
xy = reducer.fit_transform(corpus_vecs)  # この 2D 空間を以降のセルで再利用する


# %%
# ── プロット（QID / hits は §2 のセルで計算済みのものを使う）──
q_xy = reducer.transform(np.asarray(qvec, dtype=np.float32)[None, :])[0]
hit_keys = {(h["source"], h["chunk_index"]) for h in hits}
is_expected = np.array([p.get("source") in expected for p in payloads])
is_hit = np.array([(p.get("source"), p.get("chunk_index")) in hit_keys for p in payloads])

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(xy[:, 0], xy[:, 1], s=6, c="#dddddd", linewidths=0, label="全チャンク")
ax.scatter(xy[is_expected, 0], xy[is_expected, 1], s=30, c="#2e8b57",
           linewidths=0, label="正解記事のチャンク")
ax.scatter(xy[is_hit, 0], xy[is_hit, 1], s=95, facecolors="none",
           edgecolors="#d62728", linewidths=1.8, label=f"top-{config.TOP_K} ヒット")
ax.scatter(q_xy[0], q_xy[1], s=280, marker="*", c="#111111", zorder=5, label="クエリ")

# 正解記事ごとに重心へ記事名（slug末尾）を添える
for src in sorted(expected):
    m = np.array([p.get("source") == src for p in payloads])
    if m.any():
        ax.annotate(src.split("_")[-1], (xy[m, 0].mean(), xy[m, 1].mean()),
                    fontsize=9, color="#1a5e38", weight="bold")

ax.set_title(f"[{row['id']}] {row['question'][:46]}…  match_type={row['match_type']}")
ax.legend(loc="best")
ax.set_xticks([])
ax.set_yticks([])
plt.show()


# %% [markdown]
# ## 5. 拡張クエリ（HyDE）を可視化 — 元クエリ vs 仮の回答で膨らませたクエリ
#
# 参考資料の図（赤 X = 元クエリ / オレンジ X = 拡張クエリ / 緑 = 検索された doc）と同じ発想。
# LLM に**資料なしで仮の回答（hypothetical answer）を書かせ**、元質問と連結して埋め込む。
# 拡張後のクエリ点（オレンジ）が正解記事（緑）へ寄れば、クエリ拡張が効く見込み。
# AND質問（structure recall 0.35）に効くかの事前診断に使う。

# %%
from music_rag import llm as llm_module  # noqa: E402

hypo = llm_module.explain(row["question"], [], None)  # 資料なし＝仮の回答を書かせる(HyDE)
augmented = row["question"] + "\n" + hypo
aug_vec = embedder.embed_query(augmented)

orig_xy = reducer.transform(np.asarray(qvec, dtype=np.float32)[None, :])[0]
aug_xy = reducer.transform(np.asarray(aug_vec, dtype=np.float32)[None, :])[0]

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(xy[:, 0], xy[:, 1], s=6, c="#dddddd", linewidths=0, label="全チャンク")
ax.scatter(xy[is_expected, 0], xy[is_expected, 1], s=30, c="#2e8b57",
           linewidths=0, label="正解記事のチャンク")
ax.scatter(orig_xy[0], orig_xy[1], s=260, marker="X", c="#d62728",
           zorder=5, label="元クエリ")
ax.scatter(aug_xy[0], aug_xy[1], s=260, marker="X", c="#ff8c00",
           zorder=5, label="拡張クエリ(HyDE)")
for src in sorted(expected):
    m = np.array([p.get("source") == src for p in payloads])
    if m.any():
        ax.annotate(src.split("_")[-1], (xy[m, 0].mean(), xy[m, 1].mean()),
                    fontsize=9, color="#1a5e38", weight="bold")

ax.set_title(f"[{row['id']}] 元クエリ vs 拡張クエリ — 正解記事(緑)に近づくか")
ax.legend(loc="best")
ax.set_xticks([])
ax.set_yticks([])
plt.show()
print("hypothetical answer (先頭200字):\n", hypo[:200])
