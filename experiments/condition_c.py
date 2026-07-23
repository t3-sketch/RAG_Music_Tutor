"""条件C: dense + sparse + query expansion

課題: A・B の効果が独立に足し算されるか、干渉するか不明。
仮説: 相補的（QE が意味側、sparse が語彙側を拾う）なら C が最良になる。
根拠: 干渉の可能性もある（QE が付加した語が sparse の重み分布を薄める等）。
  要因計画だからこそ、この相互作用を数値で答えられる。
結果: TODO（未計測）

主指標（採否判断）: 全体 recall@5 ＋ 相互作用の符号
  相互作用 = (C − Base) − [(A − Base) + (B − Base)]
副指標: match_type / notation_variant / source の層別すべて

前提: Qdrant collection `music_theory_hybrid`（ingest は本ランナー外）。
使い方:
  uv run python experiments/condition_c.py
  uv run python experiments/condition_c.py --limit 3
"""
from __future__ import annotations

from retrieval_exp_common import parse_limit_args, run_condition


def main() -> None:
    args = parse_limit_args(__doc__ or "")
    run_condition(name="c", use_sparse=True, use_qe=True, limit=args.limit)


if __name__ == "__main__":
    main()
