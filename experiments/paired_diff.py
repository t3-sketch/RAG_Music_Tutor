"""2戦略の paired difference 分析（fixed vs structure など）。

なぜ paired か（Anthropic "Adding Error Bars to Evals" 2024）:
- 同じ質問セットを両戦略に流しているので、質問ごとの差分を取ると
  「質問の難易度由来の分散」が丸ごと消え、「戦略の差」だけが残る。
- 集計平均の比較（0.707 vs 0.662 など）より遥かに感度が高い。n=33でも使える。

やること:
- scores_YYYYMMDD.json（evaluation.main の出力）を読む
- 2戦略を id で突き合わせ、per-question の差分 d = metric(B) - metric(A) を出す
- 改善 / 悪化 / 不変の件数、平均差、平均差の bootstrap 95%CI、
  Wilcoxon 符号順位検定 / 符号検定の p値を報告する
- 差が大きく動いた質問を明細で出す（どの質問が効いたか）

使い方:
  uv run python experiments/paired_diff.py                # 既定 scores_今日.json, fixed→structure, recall
  uv run python experiments/paired_diff.py --scores data/eval/scores_20260713.json \
      --a fixed --b structure --metric recall
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from music_rag import config

METRIC_KEYS = {
    "recall": "recall",
    "strict_hit": "strict_hit",     # bool → 0/1
    "mrr": "reciprocal_rank",
}


def _index_by_id(result: dict) -> dict[str, dict]:
    return {r["id"]: r for r in result["per_question"]}


def paired_diff(scores: dict, a: str, b: str, metric: str) -> dict:
    if a not in scores or b not in scores:
        raise SystemExit(f"戦略 {a} / {b} が scores に無い（あるのは {list(scores)}）")
    key = METRIC_KEYS[metric]
    ra, rb = _index_by_id(scores[a]), _index_by_id(scores[b])
    ids = [i for i in ra if i in rb]  # 両方にある質問だけ

    rows = []
    for i in ids:
        va = float(ra[i][key])
        vb = float(rb[i][key])
        rows.append({
            "id": i,
            "a": va,
            "b": vb,
            "d": vb - va,
            "match_type": ra[i].get("match_type"),
            "source": ra[i].get("source"),
            "question": ra[i].get("question", "")[:40],
        })

    d = np.array([r["d"] for r in rows])
    n = len(d)
    improved = int((d > 0).sum())
    worsened = int((d < 0).sum())
    unchanged = int((d == 0).sum())

    # 平均差の bootstrap 95%CI（差分を復元抽出）
    rng = np.random.default_rng(42)
    boot = np.array([
        rng.choice(d, size=n, replace=True).mean() for _ in range(10000)
    ])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    # 検定: 変化のあった質問での符号順位検定（Wilcoxon）。
    # 全部tieだと検定不能なので、その場合は符号検定にフォールバック。
    nonzero = d[d != 0]
    if len(nonzero) == 0:
        test_name, pval = "変化なし（検定不能）", 1.0
    else:
        try:
            _, pval = stats.wilcoxon(d[d != 0])
            test_name = "Wilcoxon 符号順位検定"
        except ValueError:
            # 符号検定（二項検定）にフォールバック
            k = improved
            pval = stats.binomtest(k, improved + worsened, 0.5).pvalue
            test_name = "符号検定（二項）"

    return {
        "a": a, "b": b, "metric": metric, "n": n,
        "mean_a": float(np.mean([r["a"] for r in rows])),
        "mean_b": float(np.mean([r["b"] for r in rows])),
        "mean_diff": float(d.mean()),
        "ci95": ci,
        "improved": improved, "worsened": worsened, "unchanged": unchanged,
        "test_name": test_name, "pval": float(pval),
        "rows": rows,
    }


def print_report(res: dict) -> None:
    a, b, m = res["a"], res["b"], res["metric"]
    print(f"=== paired diff: {a} → {b}  /  metric={m}  (n={res['n']}) ===\n")
    print(f"  mean({a}) = {res['mean_a']:.4f}")
    print(f"  mean({b}) = {res['mean_b']:.4f}")
    print(f"  平均差 (b - a) = {res['mean_diff']:+.4f}")
    lo, hi = res["ci95"]
    crosses = lo <= 0 <= hi
    print(f"  95%CI(平均差) = [{lo:+.4f}, {hi:+.4f}]  "
          f"{'← 0をまたぐ（有意でない）' if crosses else '← 0を含まない（有意）'}")
    print(f"  改善 {res['improved']} / 悪化 {res['worsened']} / 不変 {res['unchanged']}")
    print(f"  {res['test_name']}: p = {res['pval']:.4f}\n")

    # 動いた質問の明細（|d|>0 のみ、差の大きい順）
    moved = sorted([r for r in res["rows"] if r["d"] != 0],
                   key=lambda r: -abs(r["d"]))
    if moved:
        print("  ── 差が出た質問（b-a）──")
        for r in moved:
            arrow = "↑" if r["d"] > 0 else "↓"
            print(f"    {arrow} {r['d']:+.3f}  [{r['match_type']:6s}/{r['source'][:6]}] "
                  f"{r['id']:12s} {r['question']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path,
                   default=config.EVAL_DIR / "scores_20260713.json")
    p.add_argument("--a", default="fixed", help="基準戦略")
    p.add_argument("--b", default="structure", help="比較戦略")
    p.add_argument("--metric", default="recall", choices=list(METRIC_KEYS))
    p.add_argument("--all-metrics", action="store_true",
                   help="recall / strict_hit / mrr を全部出す")
    args = p.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    metrics = list(METRIC_KEYS) if args.all_metrics else [args.metric]
    for i, m in enumerate(metrics):
        res = paired_diff(scores, args.a, args.b, m)
        print_report(res)
        if i < len(metrics) - 1:
            print()


if __name__ == "__main__":
    main()
