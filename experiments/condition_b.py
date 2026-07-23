"""条件B: dense + query expansion

課題: 度数表記・記号ゆれ ＋ 口語質問の語彙不足。
仮説: 検索前に LLM で質問を書き換え（度数表記の正規化・同義語の付加）ることで
  dense 単独でも正解チャンクに近づく。
根拠: 「5-1」→「V-I ドミナントモーション 正格終止」のような展開は、コーパス側の
  語彙にクエリを寄せる操作であり、埋め込み空間上の距離を直接縮める。
トレードオフ: クエリごとに LLM 呼び出し1回。効果が A と同等なら A を採用すべき
  （A はクエリ時コストゼロ）。
結果: TODO（未計測）

主指標（採否判断）: notation_variant タグ付き質問の recall@5
副指標: 全体 recall@5、MRR ＋ QE後のクエリ文の質的確認（expanded_query を JSON に保存）

前提: 既存 collection `music_theory_structure`（Base と同じ）。
使い方:
  uv run python experiments/condition_b.py
  uv run python experiments/condition_b.py --limit 3
"""
from __future__ import annotations

from retrieval_exp_common import parse_limit_args, run_condition


def main() -> None:
    args = parse_limit_args(__doc__ or "")
    run_condition(name="b", use_sparse=False, use_qe=True, limit=args.limit)


if __name__ == "__main__":
    main()
