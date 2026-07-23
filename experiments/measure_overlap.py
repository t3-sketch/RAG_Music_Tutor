"""生成回答が retrieval 元の原文をどれだけそのまま再現しているかを測る。

公開デモで著作物を実質的に再配布していないことを、印象でなく数字で示すための計測。
指標は「回答と取得チャンクの最長共通部分文字列（連続一致）の文字数」。
日本語なので文字単位で見る（形態素解析は不要／過剰）。

目安: 30〜50字 = 用語や定型句の一致。150字超 = 原文の写しを疑う。

  uv run python experiments/measure_overlap.py            # 既定20問
  uv run python experiments/measure_overlap.py --n 66     # 全問（LLMコール66回、RPD注意）
"""
import argparse
import json
from difflib import SequenceMatcher

from music_rag import config
from music_rag.query_pipeline import answer_query

EVAL_SET = config.DATA_DIR / "eval" / "eval_set_merged.json"


def longest_common_run(a: str, b: str) -> tuple[int, str]:
    """a と b の最長共通部分文字列の (長さ, 中身) を返す。"""
    # autojunk は長い文字列で頻出文字を勝手に無視するため、照合漏れを避けて無効化する
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size, a[m.a:m.a + m.size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="先頭から何問流すか")
    ap.add_argument("--out", default=None, help="per-question 結果の保存先 JSON")
    args = ap.parse_args()

    questions = json.loads(EVAL_SET.read_text(encoding="utf-8"))[: args.n]
    rows = []

    for i, q in enumerate(questions, 1):
        result = answer_query(q["question"])
        answer = result["answer"]
        # 取得チャンクのうち最も長く一致したものを、その回答の代表値とする
        size, text = max(
            (longest_common_run(answer, c["text"]) for c in result["contexts"]),
            default=(0, ""),
        )
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answer_chars": len(answer),
            "longest_run": size,
            "ratio": round(size / len(answer), 3) if answer else 0.0,
            "matched": text,
        })
        print(f"[{i}/{len(questions)}] {q['id']}: {size}字 ({rows[-1]['ratio']:.1%} of answer)")

    runs = sorted(r["longest_run"] for r in rows)
    worst = max(rows, key=lambda r: r["longest_run"])
    print(f"\nn={len(rows)}  中央値 {runs[len(runs) // 2]}字 / 最大 {runs[-1]}字")
    print(f"最長一致（{worst['id']}）: {worst['matched'][:120]!r}")

    if args.out:
        out = config.DATA_DIR / "eval" / args.out
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out}")


def _selfcheck() -> None:
    assert longest_common_run("abcXYZdef", "123XYZ456")[1] == "XYZ"
    assert longest_common_run("abc", "xyz")[0] == 0
    # 部分一致が複数あるときは最長側を採る
    assert longest_common_run("aa-bbbb-cc", "bbbb aa")[1] == "bbbb"
    print("selfcheck ok")


if __name__ == "__main__":
    import sys
    _selfcheck() if "--selfcheck" in sys.argv else main()
