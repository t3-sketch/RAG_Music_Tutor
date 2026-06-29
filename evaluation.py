"""retrieval 評価モジュール（hit-rate / MRR）。

責務:
- eval set（data/eval/questions.json）を読む
- 各質問を embedder → retriever に通して hits を得る
- hit-rate@k / MRR を計算する
- 複数の chunking 戦略（config.COLLECTIONS）を比較する
- スコアを data/eval/scores_YYYYMMDD.json に保存する

設計上の立場:
- evaluation.py は main.py と同じ「接着剤」。embedder / retriever を直接呼ぶ。
- custom_types は import しない（素の dict / primitive で扱う）。
- LLM を使わない（hit-rate / MRR は retrieval のみ）。RAGAS は別途（下部にコメントで保留）。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import config
import embedder
import retriever

EVAL_PATH = Path(__file__).resolve().parent / "data" / "eval" / "questions.json"
SCORES_DIR = Path(__file__).resolve().parent / "data" / "eval"


# ── eval set 読み込み ───────────────────────────
def load_eval_set() -> list[dict]:
    """questions.json を読む。各問は question / expected_source を持つ。"""
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    # hit-rate に最低限必要なフィールドだけ検証する
    for i, row in enumerate(data):
        if "question" not in row or "expected_source" not in row:
            raise ValueError(f"row {i}: question / expected_source が必要です")
    return data


# ── 1問を retrieve ──────────────────────────────
def retrieve_sources(question: str, top_k: int, collection: str) -> list[str]:
    """1問を embed → search し、ヒットした source のリストを順位順で返す。"""
    vec = embedder.embed_query(question)
    hits = retriever.search(vec, top_k=top_k, collection=collection)
    return [h["source"] for h in hits]


# ── 指標 ────────────────────────────────────────
def _hit_at_k(retrieved: list[str], expected: str, k: int) -> bool:
    """top-k に expected が含まれるか。"""
    return expected in retrieved[:k]


def _reciprocal_rank(retrieved: list[str], expected: str) -> float:
    """expected が最初に現れた順位の逆数。出なければ 0。"""
    for rank, source in enumerate(retrieved, start=1):
        if source == expected:
            return 1.0 / rank
    return 0.0


# ── 1 collection を評価 ─────────────────────────
def evaluate_retrieval(
    eval_set: list[dict],
    collection: str,
    k: int = config.TOP_K,
) -> dict:
    """1つの collection について hit-rate@k と MRR を計算する。"""
    hits = 0
    rr_sum = 0.0
    per_question = []

    for row in eval_set:
        retrieved = retrieve_sources(row["question"], top_k=k, collection=collection)
        expected = row["expected_source"]

        is_hit = _hit_at_k(retrieved, expected, k)
        rr = _reciprocal_rank(retrieved, expected)

        hits += int(is_hit)
        rr_sum += rr

        per_question.append({
            "question": row["question"],
            "expected": expected,
            "retrieved": retrieved,
            "hit": is_hit,
            "reciprocal_rank": round(rr, 4),
        })

    n = len(eval_set)
    return {
        "collection": collection,
        "n": n,
        "k": k,
        "hit_rate": round(hits / n, 4) if n else 0.0,
        "mrr": round(rr_sum / n, 4) if n else 0.0,
        "per_question": per_question,
    }


# ── 全戦略を比較 ────────────────────────────────
def main() -> None:
    eval_set = load_eval_set()
    print(f"loaded {len(eval_set)} questions\n")

    all_scores = {}
    for strategy, collection in config.COLLECTIONS.items():
        # collection が存在しない戦略はスキップ（未 ingestion）
        result = evaluate_retrieval(eval_set, collection=collection)
        all_scores[strategy] = result

        print(f"[{strategy}] collection={collection}")
        print(f"  hit_rate@{result['k']} = {result['hit_rate']}")
        print(f"  mrr                = {result['mrr']}")
        print()

    # 日付付きで保存（前後比較のため）
    out = SCORES_DIR / f"scores_{date.today():%Y%m%d}.json"
    out.write_text(json.dumps(all_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved → {out}")


# ════════════════════════════════════════════════
#  RAGAS による生成層評価（hit-rate で方向性を掴んだ後の節目で使う）
#  LLM を大量消費するため、普段は回さない。
#  評価モデルは RPD の大きい Gemini 3.1 Flash Lite（RPD 500）を想定。
# ════════════════════════════════════════════════
# def evaluate_generation(eval_set, collection):
#     """RAGAS で ContextRecall / ContextPrecision / Faithfulness /
#     AnswerRelevancy / AnswerCorrectness を計算する。
#
#     必要な行（RAGAS 0.4.x の新スキーマ）:
#       user_input / retrieved_contexts / response / reference
#
#     - retrieved_contexts: retriever.search の各 hit の text
#     - response: llm.explain で生成（本番と同じ生成層）
#     - reference: eval set の ground_truth
#
#     実装は次のセッションで。Gemini レート制限の設計（生成層と評価層で
#     モデルを分ける）を踏まえてから。
#     """
#     ...


if __name__ == "__main__":
    main()