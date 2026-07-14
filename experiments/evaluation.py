"""retrieval / 生成 評価モジュール（hit-rate・MRR・RAGAS）。

責務:
- eval set（data/eval/questions.json）を読む
- 各質問を embedder → retriever に通して hits を得る
- hit-rate@k / MRR を計算する（retrieval層、LLM不使用）
- RAGAS で Faithfulness / AnswerRelevancy / AnswerCorrectness /
  ContextPrecision / ContextRecall を計算する（生成層、LLM使用・節目のみ）
- 複数の chunking 戦略（config.COLLECTIONS）を比較する
- スコアを data/eval/scores_YYYYMMDD.json / ragas_*.json / ragas_*.csv に保存する

設計上の立場:
- evaluation.py は main.py と同じ「接着剤」。embedder / retriever / llm を直接呼ぶ。
- custom_types は import しない（素の dict / primitive で扱う）。
- hit-rate / MRR は retrieval のみ（LLM不使用、普段から回せる）。
- RAGAS は LLM を大量消費するため、普段は回さない。節目でのみ実行する。
  生成層（llm.explain）と評価層（judge LLM）でモデルを分けて、
  片方のレート制限に評価全体が引きずられないようにする。
  1問ごとにチェックポイントへ保存し、途中で落ちても再実行で続きから
  再開できるようにする（同じ collection の再実行時に既に終わった質問はスキップ）。
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
from datetime import date
from pathlib import Path

from music_rag import config
from music_rag import embedder
from music_rag import retriever
from music_rag import llm as llm_module  # 生成層（本番と同じ NVIDIA 呼び出し）

EVAL_PATH = config.EVAL_DIR / "eval_set_merged.json"
SCORES_DIR = config.EVAL_DIR

# RAGASの評価者（judge）モデル。生成層（NVIDIA・meta/llama-3.3-70b-instruct）とは
# 意図的にプロバイダごと分離する（self-preference bias回避 + レート制限の独立）。
# Gemini を使う（config.GEMINI_MODEL、既定 gemini-3.1-flash-lite）。モデルは
# env（RAGAS_JUDGE_MODEL）で上書き可能。
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", config.GEMINI_MODEL)

# 質問間ウェイト（秒）。judge・embeddings とも Gemini 無料枠に集中するため、
# RPD枯渇（20問中11問しか回らなかった問題）を踏まえてNVIDIA分離時より余裕を持たせる。
RAGAS_SLEEP_SEC = int(os.getenv("RAGAS_SLEEP_SEC", "15"))


# ── eval set 読み込み ───────────────────────────
def load_eval_set() -> list[dict]:
    """統合 eval set（eval_set_merged.json）を読む。
    各問は question / expected_source(list) / match_type を持つ。
    experiments/build_eval_set.py で生成する。"""
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    for i, row in enumerate(data):
        if "question" not in row or "expected_source" not in row:
            raise ValueError(f"row {i}: question / expected_source が必要です")
        if not isinstance(row["expected_source"], list):
            raise ValueError(f"row {i}: expected_source は list である必要があります")
    return data


# ── 1問を retrieve ──────────────────────────────
def retrieve_sources(question: str, top_k: int, collection: str) -> list[str]:
    """1問を embed → search し、ヒットした source のリストを順位順で返す。"""
    vec = embedder.embed_query(question)
    hits = retriever.search(vec, top_k=top_k, collection=collection)
    return [h["source"] for h in hits]


def retrieve_with_text(question: str, top_k: int, collection: str) -> list[dict]:
    """1問を embed → search し、text/source/score を保持したまま返す（RAGAS用）。"""
    vec = embedder.embed_query(question)
    return retriever.search(vec, top_k=top_k, collection=collection)


# ── 指標（recall@k 主 / strict hit-rate / MRR） ──
# 多ソース対応。expected_source は常に list。
#   recall@k       : top-k に入った正解記事の割合（主指標。連続値、部分点が見える）
#   strict_hit@k   : match_type を尊重した二値。and=全記事必須 / or・single=1つでOK
#   MRR            : いずれかの正解記事が最初に現れた順位の逆数
def _recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """top-k に入った正解記事の割合（主指標）。多ソースでも連続値で測れる。"""
    if not expected:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for e in expected if e in top) / len(expected)


def _strict_hit_at_k(
    retrieved: list[str], expected: list[str], k: int, match_type: str
) -> bool:
    """match_type を尊重した二値hit。
    and → 全正解記事が top-k に揃って初めて hit。
    or / single → どれか1つ取れれば hit。"""
    if not expected:
        return False
    top = set(retrieved[:k])
    hits = [e in top for e in expected]
    return all(hits) if match_type == "and" else any(hits)


def _reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """いずれかの正解記事が最初に現れた順位の逆数。出なければ 0。"""
    exp = set(expected)
    for rank, source in enumerate(retrieved, start=1):
        if source in exp:
            return 1.0 / rank
    return 0.0


def _mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


# ── 1 collection を評価（hit-rate / MRR） ───────
def evaluate_retrieval(
    eval_set: list[dict],
    collection: str,
    k: int = config.TOP_K,
) -> dict:
    """1つの collection について recall@k（主指標）・strict hit-rate@k・MRR を計算する。
    source（silver_manual/forum）・match_type（single/and/or）・difficulty で層別集計も返す。"""
    per_question = []

    for row in eval_set:
        retrieved = retrieve_sources(row["question"], top_k=k, collection=collection)
        expected = row["expected_source"]
        match_type = row.get("match_type", "or")

        recall = _recall_at_k(retrieved, expected, k)
        is_hit = _strict_hit_at_k(retrieved, expected, k, match_type)
        rr = _reciprocal_rank(retrieved, expected)

        per_question.append({
            "id": row.get("id"),
            "question": row["question"],
            "expected": expected,
            "match_type": match_type,
            "source": row.get("source"),
            "difficulty": row.get("difficulty"),
            "reviewed": row.get("reviewed"),
            "retrieved": retrieved,
            "recall": round(recall, 4),
            "strict_hit": is_hit,
            "reciprocal_rank": round(rr, 4),
        })

    def summarize(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "recall_at_k": _mean([r["recall"] for r in rows]),
            "strict_hit_rate": _mean([float(r["strict_hit"]) for r in rows]),
            "mrr": _mean([r["reciprocal_rank"] for r in rows]),
        }

    def stratify(key: str) -> dict:
        groups: dict[str, list[dict]] = {}
        for r in per_question:
            groups.setdefault(str(r.get(key)), []).append(r)
        return {g: summarize(rs) for g, rs in sorted(groups.items())}

    overall = summarize(per_question)
    return {
        "collection": collection,
        "k": k,
        **overall,
        "by_source": stratify("source"),
        "by_match_type": stratify("match_type"),
        "by_difficulty": stratify("difficulty"),
        "per_question": per_question,
    }


# ── 全戦略を比較（hit-rate / MRR） ──────────────
def main() -> None:
    eval_set = load_eval_set()
    print(f"loaded {len(eval_set)} questions\n")

    all_scores = {}
    for strategy, collection in config.COLLECTIONS.items():
        # collection が存在しない戦略はスキップ（未 ingestion / 削除済み）
        try:
            result = evaluate_retrieval(eval_set, collection=collection)
        except Exception as e:
            print(f"[{strategy}] collection={collection} をスキップ: {e}\n")
            continue
        all_scores[strategy] = result

        print(f"[{strategy}] collection={collection}  (n={result['n']}, k={result['k']})")
        print(f"  recall@{result['k']}      = {result['recall_at_k']}")
        print(f"  strict_hit_rate = {result['strict_hit_rate']}")
        print(f"  mrr             = {result['mrr']}")
        print("  ── by match_type ──")
        for mt, sc in result["by_match_type"].items():
            print(f"    {mt:8s} n={sc['n']:2d}  recall={sc['recall_at_k']:.3f}  "
                  f"strict_hit={sc['strict_hit_rate']:.3f}  mrr={sc['mrr']:.3f}")
        print("  ── by source ──")
        for src, sc in result["by_source"].items():
            print(f"    {src:14s} n={sc['n']:2d}  recall={sc['recall_at_k']:.3f}  "
                  f"strict_hit={sc['strict_hit_rate']:.3f}  mrr={sc['mrr']:.3f}")
        print()

    # 日付付きで保存（前後比較のため）
    out = SCORES_DIR / f"scores_{date.today():%Y%m%d}.json"
    out.write_text(json.dumps(all_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved → {out}")


# ════════════════════════════════════════════════
#  RAGAS による生成層評価（hit-rate で方向性を掴んだ後の節目で使う）
#  LLM を大量消費するため、普段は回さない。
# ════════════════════════════════════════════════

def _ragas_setup():
    """RAGAS評価用のLLM/embeddingsをセットアップする。

    judge LLM は Gemini の OpenAI互換エンドポイント経由（AsyncOpenAI）。
    google-genai ネイティブクライアントはRAGASのアダプタと相性が悪いため
    （instructor/litellmアダプタが非同期判定に失敗する）、この経路を使う。
    embeddings（AnswerRelevancy / AnswerCorrectness のみ使用）も Gemini。
    max_tokens はデフォルトだと日本語＋複数statement照合で出力が途中で切れる
    （IncompleteOutputException）ため、明示的に大きめに設定する。
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from google import genai
    from ragas.embeddings import GoogleEmbeddings

    async_client = AsyncOpenAI(
        api_key=config.GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    ragas_llm = llm_factory(
        RAGAS_JUDGE_MODEL,
        provider="openai",
        client=async_client,
        max_tokens=8192,   # ← デフォルトだと途中で切れるので増やす
    )

    genai_client = genai.Client(api_key=config.GEMINI_API_KEY)
    embeddings = GoogleEmbeddings(client=genai_client, model="gemini-embedding-001")

    return ragas_llm, embeddings


def _checkpoint_path(collection: str) -> Path:
    """RAGAS途中経過の保存先。collectionごとに分ける。"""
    return SCORES_DIR / f"_ragas_checkpoint_{collection}.json"


def _load_checkpoint(collection: str) -> list[dict]:
    """既存のチェックポイントがあれば読み込む。無ければ空リスト。"""
    path = _checkpoint_path(collection)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save_checkpoint(collection: str, per_question: list[dict]) -> None:
    """1問終わるたびにチェックポイントを丸ごと書き直す（追記ではなく上書き、壊れ防止）。"""
    path = _checkpoint_path(collection)
    path.write_text(json.dumps(per_question, ensure_ascii=False, indent=2), encoding="utf-8")


def _clear_checkpoint(collection: str) -> None:
    """全問完了後、チェックポイントを削除する。"""
    path = _checkpoint_path(collection)
    if path.exists():
        path.unlink()


async def evaluate_generation(eval_set: list[dict], collection: str) -> dict:
    """RAGAS で ContextRecall / ContextPrecision / Faithfulness /
    AnswerRelevancy / AnswerCorrectness を計算する。

    - retrieved_contexts: retriever.search の各 hit の text
    - response: llm.explain で生成（本番と同じ生成層、NVIDIA・config.NVIDIA_LLM_MODEL）
    - reference: eval set の ground_truth
    - judge: RAGAS_JUDGE_MODEL（生成層とは別モデルでレート制限を分離）

    途中で落ちても、同じ collection で再実行すれば、チェックポイントに
    保存済みの質問はスキップして続きから再開する。
    """
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        AnswerCorrectness,
        ContextPrecision,
        ContextRecall,
    )

    ragas_llm, embeddings = _ragas_setup()

    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=embeddings)
    answer_correctness = AnswerCorrectness(llm=ragas_llm, embeddings=embeddings)
    context_precision = ContextPrecision(llm=ragas_llm)
    context_recall = ContextRecall(llm=ragas_llm)

    # 既存チェックポイントを読み込み、済みの質問をスキップ対象にする
    per_question = _load_checkpoint(collection)
    done_questions = {row["question"] for row in per_question}
    if done_questions:
        print(f"  チェックポイントから再開: {len(done_questions)} 問は完了済み\n")

    remaining = [row for row in eval_set if row["question"] not in done_questions]
    total = len(eval_set)

    for row in remaining:
        question = row["question"]
        reference = row.get("ground_truth", "")
        i = len(per_question) + 1

        hits = retrieve_with_text(question, top_k=config.TOP_K, collection=collection)
        contexts = [h["text"] for h in hits]

        # 本番と同じ生成層（llm.explain）で回答を作る（503等は内部でリトライ済み）
        chunks_for_llm = [{"text": h["text"], "meta": {"source": h["source"]}} for h in hits]
        response = llm_module.explain(question, chunks_for_llm, None)

        faith_result = await faithfulness.ascore(
            user_input=question, response=response, retrieved_contexts=contexts
        )
        relevancy_result = await answer_relevancy.ascore(
            user_input=question, response=response
        )
        correctness_result = await answer_correctness.ascore(
            user_input=question, response=response, reference=reference
        )
        precision_result = await context_precision.ascore(
            user_input=question, retrieved_contexts=contexts, reference=reference
        )
        recall_result = await context_recall.ascore(
            user_input=question, retrieved_contexts=contexts, reference=reference
        )

        per_question.append({
            "question": question,
            "response": response,
            "faithfulness": faith_result.value,
            "answer_relevancy": relevancy_result.value,
            "answer_correctness": correctness_result.value,
            "context_precision": precision_result.value,
            "context_recall": recall_result.value,
        })
        _save_checkpoint(collection, per_question)  # 1問ごとに保存
        print(f"  [{i}/{total}] done: {question[:30]}...")

        # レート制限（judge 40RPM / embedding RPM 100）回避。最後の問では待たない
        if i < total:
            await asyncio.sleep(RAGAS_SLEEP_SEC)

    n = len(per_question)

    def avg(key: str) -> float:
        return round(sum(q[key] for q in per_question) / n, 4) if n else 0.0

    result = {
        "collection": collection,
        "n": n,
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "answer_correctness": avg("answer_correctness"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }

    _clear_checkpoint(collection)  # 全問完了したのでチェックポイントは不要
    return result


# ── RAGAS結果を CSV に変換（Excel / Googleスプレッドシート用） ──
def export_ragas_csv(result: dict, out_path: Path | None = None) -> Path:
    """RAGAS結果(dict)を CSV に変換する。1行=1問、列=5指標+question+response。"""
    if out_path is None:
        out_path = SCORES_DIR / f"ragas_{result['collection']}_{date.today():%Y%m%d}.csv"

    fieldnames = [
        "question", "response",
        "faithfulness", "answer_relevancy", "answer_correctness",
        "context_precision", "context_recall",
    ]

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        # utf-8-sig: Excelで開いたときに日本語が文字化けしないようBOM付きにする
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["per_question"]:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"CSV saved → {out_path}")
    return out_path


def run_ragas(collection: str = "music_theory", limit: int | None = None) -> dict:
    """RAGAS評価を同期的に実行するエントリーポイント。

    limit: 動作確認用に先頭N問だけ回す場合に指定（本番は None で全問）。
    途中で落ちても、同じ collection で再実行すれば続きから再開する。
    """
    eval_set = load_eval_set()
    if limit:
        eval_set = eval_set[:limit]
        print(f"limit={limit}: 先頭 {len(eval_set)} 問のみ実行\n")

    result = asyncio.run(evaluate_generation(eval_set, collection))

    print(f"\n[RAGAS] collection={collection}")
    print(f"  faithfulness       = {result['faithfulness']}")
    print(f"  answer_relevancy   = {result['answer_relevancy']}")
    print(f"  answer_correctness = {result['answer_correctness']}")
    print(f"  context_precision  = {result['context_precision']}")
    print(f"  context_recall     = {result['context_recall']}")

    out = SCORES_DIR / f"ragas_{collection}_{date.today():%Y%m%d}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved → {out}")

    export_ragas_csv(result)

    return result


if __name__ == "__main__":
    main()