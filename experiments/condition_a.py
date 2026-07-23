"""条件A: dense + sparse（ハイブリッド検索）

課題: 度数表記・記号トークンの一致に dense が弱い。
仮説: sparse（語彙重み）の追加で表記一致の質問の recall が改善する。
根拠: BGE-M3 は dense/sparse を1パスで出力でき追加コストが最小。
  BM25系の語彙一致が記号・固有語に強いのは IR の定石。クエリ時の追加レイテンシほぼゼロ。
結果: TODO（未計測）

主指標（採否判断）: notation_variant タグ付き質問の recall@5
副指標: 全体 recall@5、MRR

前提: Qdrant collection `music_theory_hybrid`（ingest は本ランナー外）。
使い方:
  uv run python experiments/condition_a.py
  uv run python experiments/condition_a.py --limit 3
"""
from __future__ import annotations

from retrieval_exp_common import parse_limit_args, run_condition


def main() -> None:
    args = parse_limit_args(__doc__ or "")
    run_condition(name="a", use_sparse=True, use_qe=False, limit=args.limit)


if __name__ == "__main__":
    main()
