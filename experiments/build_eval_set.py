"""統合 eval set のビルダー。

2つのソースを1つの統一スキーマにまとめる:
  1. data/eval/questions.json      — 手作りsilver 20問（単一記事・記事内セクション統合型）
  2. data/eval/forum_review.json   — フォーラム由来。review_status=="reviewed" かつ
                                     answerable_standalone==True のものだけ採用

統一スキーマ（1問1dict）:
  id            : 一意なID（old_NN / forum_<topic_id>）
  question      : 質問文
  ground_truth  : 参照回答
  expected_source: list[str]  ← 常にlist。正解記事のslug（拡張子なし）
  match_type    : "single" | "and" | "or"
                  single = 正解が1記事（旧セット）
                  and    = top-kに全記事が揃って初めて正解（比較・統合質問）
                  or     = 候補のどれか1つが取れれば正解
  difficulty    : easy/medium/hard
  source        : "silver_manual" | "forum"
  reviewed      : bool  ← 正解ラベルが人手検証済みか。False=LLM機械推定のまま（未検証）。
                  スコア解釈時に必ず区別すること（未レビュー分はラベルノイズを含む）。
  topic         : 旧セットのみ（chord/melody/rhythm/prerequisite）

出力: data/eval/eval_set_merged.json

デフォルトは reviewed 済みのみ。--include-pending で未レビュー(pending)の
answerable も LLM推定ラベルのまま投入する（reviewed=False で明示タグ付け）。
forum側のレビューが進むたびに再実行すれば、reviewedになった分が自動で増える。
"""
from __future__ import annotations

import argparse
import json

from music_rag import config

OLD_PATH = config.EVAL_DIR / "questions.json"
FORUM_PATH = config.EVAL_DIR / "forum_review.json"
OUT_PATH = config.EVAL_DIR / "eval_set_merged.json"


def build(include_pending: bool = False) -> list[dict]:
    rows: list[dict] = []

    # --- 1. 旧silver 20問（すべて人手作成＝reviewed）---
    old = json.loads(OLD_PATH.read_text(encoding="utf-8"))
    for i, q in enumerate(old, start=1):
        rows.append({
            "id": f"old_{i:02d}",
            "question": q["question"],
            "ground_truth": q.get("ground_truth", ""),
            "expected_source": [q["expected_source"]],  # 単一→listに正規化
            "match_type": "single",
            "difficulty": q.get("difficulty"),
            "source": "silver_manual",
            "reviewed": True,
            "topic": q.get("topic"),
        })

    # --- 2. forum ---
    # 既定: reviewed かつ answerable のみ。
    # include_pending: 未レビュー(pending)の answerable も LLM推定ラベルのまま投入し、
    #                  reviewed=False で明示タグ付けする（正解ラベルは未検証）。
    forum = json.loads(FORUM_PATH.read_text(encoding="utf-8"))
    for tid, v in forum.items():
        if not v.get("answerable_standalone"):
            continue
        is_reviewed = v.get("review_status") == "reviewed"
        if not is_reviewed and not include_pending:
            continue
        rows.append({
            "id": f"forum_{tid}",
            "question": v["question"],
            "ground_truth": v.get("ground_truth", ""),
            "expected_source": v["expected_source"],
            "match_type": v.get("match_type") or "or",
            "difficulty": v.get("difficulty"),
            "source": "forum",
            "reviewed": is_reviewed,  # False = LLM推定ラベルのまま（未検証）
            "topic": None,
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="未レビュー(pending)のforum answerableもLLM推定ラベルのまま投入（reviewed=Falseでタグ付け）",
    )
    args = parser.parse_args()

    rows = build(include_pending=args.include_pending)
    OUT_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from collections import Counter
    print(f"wrote {len(rows)} questions → {OUT_PATH}")
    print("  source:", dict(Counter(r["source"] for r in rows)))
    print("  reviewed:", dict(Counter(r["reviewed"] for r in rows)))
    print("  match_type:", dict(Counter(r["match_type"] for r in rows)))
    print("  difficulty:", dict(Counter(r["difficulty"] for r in rows)))
    multi = [r for r in rows if len(r["expected_source"]) > 1]
    print(f"  multi-source questions: {len(multi)}")
    n_pending = sum(1 for r in rows if not r["reviewed"])
    if n_pending:
        print(f"  ⚠️  未レビュー(reviewed=False)を {n_pending} 問含む: 正解ラベルはLLM推定のまま")


if __name__ == "__main__":
    main()
