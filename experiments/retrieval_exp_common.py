"""検索層 2×2 実験（dense/sparse × QE）の共有処理。

条件 A/B/C の評価ランナー（condition_*.py）から呼ばれる。
本番の embedder / retriever / query_pipeline は変更しない（実験隔離）。

- dense: music_rag.embedder + retriever.search
- hybrid: BGE-M3 sparse + Qdrant Prefetch/RRF（collection = music_theory_hybrid）
- QE: gemini-3.1-flash-lite で expansion（rewrite ではなく追記）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import date
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient, models

from music_rag import config, embedder, retriever

# evaluation.py と同ディレクトリ実行想定（uv run python experiments/condition_*.py）
from evaluation import (  # noqa: E402
    _mean,
    _recall_at_k,
    _reciprocal_rank,
    _strict_hit_at_k,
    load_eval_set,
)

HYBRID_COLLECTION = "music_theory_hybrid"
DENSE_COLLECTION = config.COLLECTIONS["structure"]  # music_theory_structure
# QE 専用の model 切替（config.GEMINI_MODEL は RAGAS judge 兼用なので分離する）。
# env QE_MODEL で上書き。既定は config.GEMINI_MODEL（gemini-3.1-flash-lite）。
QE_MODEL = os.getenv("QE_MODEL", config.GEMINI_MODEL)
_QE_SLUG = QE_MODEL.replace("/", "_")  # ファイル名タグ（cache / 出力をモデル別に）
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
PREFETCH_LIMIT = 50
EVAL_SET_NAME = "eval_set_merged.json"

EXPAND_SYSTEM = """あなたは日本語の音楽理論に詳しい検索クエリ拡張アシスタントです。
ユーザーの質問文を書き換えず、そのまま残したうえで、検索に効く同義語・正規化表記を追記してください。

ルール:
- 元の質問文は1行目にそのまま残す（削除・言い換え禁止）。
- 2行目以降に、度数表記の正規化（例: 「5-1」→「V-I」「ドミナントモーション」「正格終止」）や
  同義の理論用語をスペース区切りで追記する。
- 解説文や長文は書かない。追記は短い語・フレーズのみ。
- 出力はその2ブロックだけ（前置き・後書きなし）。
"""


# ── QE ──────────────────────────────────────────
@lru_cache(maxsize=1)
def _qe_client():
    from openai import OpenAI

    # QE 展開は数秒で返るはず。無料枠がまれに stall するので短い timeout で
    # 早く諦めて retry する（config.LLM_TIMEOUT_SEC=90 だと1回の hang で9分無駄）。
    return OpenAI(
        api_key=config.GEMINI_API_KEY,
        base_url=config.GEMINI_BASE_URL,
        timeout=25.0,
    )


# Gemini 無料枠は flash-lite 15 RPM。66問を連射すると 429 になるので
# 呼び出し間隔を空け（12 RPM 相当）、429/timeout は数回リトライする。
# 展開結果は cache に残す: 失敗した run の再開でも RPD を無駄にせず、
# 条件C は条件B と同じ問なので QE 呼び出しをまるごと再利用できる。
_QE_MIN_INTERVAL = 5.0
_last_qe_call = 0.0
_QE_CACHE_FILE = (
    Path(__file__).resolve().parent / "_hybrid_scratch" / f"qe_cache_{_QE_SLUG}.json"
)


@lru_cache(maxsize=1)
def _qe_cache() -> dict:
    if _QE_CACHE_FILE.exists():
        return json.loads(_QE_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_qe_cache() -> None:
    _QE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _QE_CACHE_FILE.write_text(
        json.dumps(_qe_cache(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _qe_create(question: str) -> str:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    retryable = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    global _last_qe_call
    for attempt in range(6):
        wait = _QE_MIN_INTERVAL - (time.monotonic() - _last_qe_call)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = _qe_client().chat.completions.create(
                model=QE_MODEL,
                messages=[
                    {"role": "system", "content": EXPAND_SYSTEM},
                    {"role": "user", "content": question},
                ],
                temperature=0,
            )
            _last_qe_call = time.monotonic()
            return (resp.choices[0].message.content or "").strip()
        except retryable as e:
            _last_qe_call = time.monotonic()
            m = re.search(r"retry in ([\d.]+)s", str(e))
            delay = float(m.group(1)) + 1 if m else min(10 * (attempt + 1), 45)
            print(f"    {type(e).__name__}: {delay:.0f}s 待機して再試行（{attempt + 1}/6）")
            time.sleep(delay)
    return None  # リトライ上限: 呼び出し側で raw question にフォールバック


def expand_query(question: str) -> str:
    """検索前の query expansion（元文保持 + 正規化・同義語の追記）。cache 優先。

    QE がリトライ上限まで失敗したら raw question を返す（1問の stall で
    run 全体を落とさない）。失敗は cache しない → 後の再実行でやり直せる。
    """
    cache = _qe_cache()
    if question in cache:
        return cache[question]
    text = _qe_create(question)
    if text is None:
        print(f"    QE 失敗 → raw question で検索（未cache・再実行で再試行）")
        return question
    # モデルが元文を落とした場合の保険: 先頭に元質問を戻す
    if question not in text:
        text = f"{question}\n{text}"
    cache[question] = text
    _save_qe_cache()
    return text


# ── dense / hybrid retrieve（実験専用）──────────
def retrieve_dense(question: str, top_k: int, collection: str) -> list[str]:
    vec = embedder.embed_query(question)
    hits = retriever.search(vec, top_k=top_k, collection=collection)
    return [h["source"] for h in hits]


@lru_cache(maxsize=1)
def _bge_m3():
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(config.EMBED_MODEL, use_fp16=False)


def _encode_dense_sparse(text: str) -> tuple[list[float], models.SparseVector]:
    """BGE-M3 から dense + lexical_weights を1パスで取る（実験専用・本番 embedder 不変）。"""
    out = _bge_m3().encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = [float(x) for x in out["dense_vecs"][0]]
    weights = out["lexical_weights"][0]
    # FlagEmbedding: dict[int|str, float]
    indices: list[int] = []
    values: list[float] = []
    for k, v in weights.items():
        indices.append(int(k))
        values.append(float(v))
    return dense, models.SparseVector(indices=indices, values=values)


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def retrieve_hybrid(question: str, top_k: int, collection: str) -> list[str]:
    """named vectors (dense + sparse) を Prefetch → RRF で融合して source 列を返す。"""
    dense, sparse = _encode_dense_sparse(question)
    response = _qdrant().query_points(
        collection_name=collection,
        prefetch=[
            models.Prefetch(
                query=dense,
                using=DENSE_VECTOR_NAME,
                limit=PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=sparse,
                using=SPARSE_VECTOR_NAME,
                limit=PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [(p.payload or {}).get("source", "") for p in response.points]


def _require_collection(collection: str, *, need_hybrid: bool) -> None:
    try:
        client = _qdrant()
        exists = client.collection_exists(collection)
    except Exception as e:
        raise SystemExit(
            f"Qdrant に接続できない（URL={config.QDRANT_URL}）: {e}\n"
            f"  → `docker compose up -d qdrant` で起動してから再実行。"
        ) from e
    if exists:
        return
    if need_hybrid:
        raise SystemExit(
            f"collection '{collection}' が無い。"
            f"条件 A/C は hybrid ingest（dense+sparse の再 upsert）が前提。"
            f"ingest は本ランナーのスコープ外。先に `{HYBRID_COLLECTION}` を作成してから再実行。"
        )
    raise SystemExit(f"collection '{collection}' が無い（Qdrant URL={config.QDRANT_URL}）")


# ── 採点ループ ──────────────────────────────────
def evaluate_condition(
    eval_set: list[dict],
    *,
    use_sparse: bool,
    use_qe: bool,
    k: int = config.TOP_K,
) -> dict:
    collection = HYBRID_COLLECTION if use_sparse else DENSE_COLLECTION
    _require_collection(collection, need_hybrid=use_sparse)

    retrieve_fn = retrieve_hybrid if use_sparse else retrieve_dense
    per_question: list[dict] = []

    for i, row in enumerate(eval_set, start=1):
        question = row["question"]
        expanded = expand_query(question) if use_qe else None
        query_for_search = expanded if expanded is not None else question

        retrieved = retrieve_fn(query_for_search, top_k=k, collection=collection)
        expected = row["expected_source"]
        match_type = row.get("match_type", "or")

        recall = _recall_at_k(retrieved, expected, k)
        is_hit = _strict_hit_at_k(retrieved, expected, k, match_type)
        rr = _reciprocal_rank(retrieved, expected)

        entry = {
            "id": row.get("id"),
            "question": question,
            "expected": expected,
            "match_type": match_type,
            "source": row.get("source"),
            "difficulty": row.get("difficulty"),
            "reviewed": row.get("reviewed"),
            "notation_variant": row.get("notation_variant"),
            "retrieved": retrieved,
            "recall": round(recall, 4),
            "strict_hit": is_hit,
            "reciprocal_rank": round(rr, 4),
        }
        if expanded is not None:
            entry["expanded_query"] = expanded
        per_question.append(entry)
        print(f"  [{i}/{len(eval_set)}] recall={recall:.2f}  {question[:40]}...")

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
            val = r.get(key)
            if val is None:
                continue
            groups.setdefault(str(val), []).append(r)
        return {g: summarize(rs) for g, rs in sorted(groups.items())}

    overall = summarize(per_question)
    result = {
        "collection": collection,
        "k": k,
        **overall,
        "by_source": stratify("source"),
        "by_match_type": stratify("match_type"),
        "by_difficulty": stratify("difficulty"),
        "per_question": per_question,
    }
    by_nv = stratify("notation_variant")
    if by_nv:
        result["by_notation_variant"] = by_nv
    return result


def run_condition(
    name: str,
    *,
    use_sparse: bool,
    use_qe: bool,
    limit: int | None = None,
    k: int = config.TOP_K,
) -> dict:
    """条件 name（a/b/c）を評価して scores JSON を保存する。"""
    eval_set = load_eval_set()
    if limit is not None:
        eval_set = eval_set[:limit]
        print(f"limit={limit}: 先頭 {len(eval_set)} 問のみ\n")

    print(
        f"[condition_{name}] sparse={use_sparse} qe={use_qe} "
        f"collection={'hybrid' if use_sparse else 'structure'} n={len(eval_set)}\n"
    )
    scored = evaluate_condition(
        eval_set, use_sparse=use_sparse, use_qe=use_qe, k=k
    )

    key = f"condition_{name}"
    payload = {
        "meta": {
            "condition": name,
            "use_sparse": use_sparse,
            "use_qe": use_qe,
            "collection": scored["collection"],
            "fusion": "rrf" if use_sparse else None,
            "qe_model": QE_MODEL if use_qe else None,
            "k": k,
            "eval_set": EVAL_SET_NAME,
        },
        # paired_diff.py が scores[key]["per_question"] を読める形
        key: scored,
    }

    # QE 条件はモデル名をファイル名に入れて 3.1/3.5 を並存させる（A は use_qe=False で無タグ）
    suffix = f"_{_QE_SLUG}" if use_qe else ""
    out = config.EVAL_DIR / f"scores_condition_{name}{suffix}_{date.today():%Y%m%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_out = _write_csv(f"{name}{suffix}", scored["per_question"])

    print(f"\n[{key}] n={scored['n']} k={scored['k']}")
    print(f"  recall@{scored['k']}      = {scored['recall_at_k']}")
    print(f"  strict_hit_rate = {scored['strict_hit_rate']}")
    print(f"  mrr             = {scored['mrr']}")
    print("  ── by match_type ──")
    for mt, sc in scored["by_match_type"].items():
        print(
            f"    {mt:8s} n={sc['n']:2d}  recall={sc['recall_at_k']:.3f}  "
            f"strict_hit={sc['strict_hit_rate']:.3f}  mrr={sc['mrr']:.3f}"
        )
    if "by_notation_variant" in scored:
        print("  ── by notation_variant ──")
        for nv, sc in scored["by_notation_variant"].items():
            print(
                f"    {nv:8s} n={sc['n']:2d}  recall={sc['recall_at_k']:.3f}  "
                f"strict_hit={sc['strict_hit_rate']:.3f}  mrr={sc['mrr']:.3f}"
            )
    print(f"saved → {out}")
    print(f"saved → {csv_out}")
    return payload


def _write_csv(name: str, per_question: list[dict]) -> Path:
    """per-question 結果を CSV で保存（list 列は | 区切りに平坦化）。"""
    cols = [
        "id", "question", "match_type", "source", "difficulty",
        "notation_variant", "recall", "strict_hit", "reciprocal_rank",
        "expected", "retrieved", "expanded_query",
    ]
    out = config.EVAL_DIR / f"scores_condition_{name}_{date.today():%Y%m%d}.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in per_question:
            row = dict(r)
            row["expected"] = "|".join(r.get("expected") or [])
            row["retrieved"] = "|".join(r.get("retrieved") or [])
            w.writerow(row)
    return out


def parse_limit_args(description: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="先頭 N 問だけ評価（動作確認用）",
    )
    return ap.parse_args()
