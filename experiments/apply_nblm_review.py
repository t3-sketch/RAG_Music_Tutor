"""NotebookLM のレビュー出力（out_part*.json）を forum_review.json に反映する。

前回(24問)の教訓:
- NotebookLM は実在しない slug を出すことがある → data/raw に無い slug は必ず弾く。
- 「情報が省略されていて検証不能」系の EXCLUDE は"読めなかった"だけで判定として無効
  → 保留(pending据え置き)にして人手で見る。
- 候補外の slug を出すことがあるが、実在＆内容一致なら元抽出より正しいことが多い（採用可）。

使い方:
  uv run python experiments/apply_nblm_review.py           # dry-run（何が起きるか表示のみ）
  uv run python experiments/apply_nblm_review.py --apply   # forum_review.json を実際に更新
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

from music_rag import config

NBLM_DIR = config.EVAL_DIR / "nblm"
FORUM_PATH = config.EVAL_DIR / "forum_review.json"
RAW_DIR = config.RAW_DIR

# 「読めなかった」ことを理由にした EXCLUDE を検出する語（判定として無効）
UNREADABLE_PAT = re.compile(
    r"省略|検証不能|読み取れ|提示され(てい)?ない|情報がない|記載がされていない|見え(ない|ません)"
)

ROW = re.compile(r"^\|\s*(\d{4,6})\s*\|([^|]*)\|([^|]*)\|([^|]*)\|(.*)$")


def known_slugs() -> set[str]:
    return {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(str(RAW_DIR / "*.json"))
    }


def parse_answers() -> dict[str, dict]:
    """out_part*.json の answer 本文から表の行を拾う。"""
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(str(NBLM_DIR / "out_part*.json"))):
        try:
            answer = json.load(open(path))["answer"]
        except Exception as e:  # 出力が壊れている part はスキップして報告
            print(f"  !! {os.path.basename(path)} を読めません: {e}")
            continue
        # 表が改行で折り返されることがあるので、行内改行を潰してから行単位で見る
        flat = re.sub(r"\n(?!\s*\|)", " ", answer)
        for line in flat.splitlines():
            m = ROW.match(line.strip())
            if not m:
                continue
            tid, verdict, slugs_raw, mt, reason = (x.strip() for x in m.groups())
            if tid in out:
                continue  # 同じ問が複数partに出たら先勝ち
            slugs = [
                s.strip().strip("`")
                for s in slugs_raw.replace("、", ",").split(",")
                if s.strip() and s.strip() not in {"なし", "-"}
            ]
            out[tid] = {
                "verdict": verdict.upper(),
                "slugs": slugs,
                "match_type": mt,
                "reason": reason.strip(),
                "src": os.path.basename(path),
            }
    return out


def normalize_mt(mt: str, slugs: list[str]) -> str:
    if "and" in mt.lower():
        return "and"
    if "or" in mt.lower():
        return "or"
    return "and" if len(slugs) > 1 else "or"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="実際に forum_review.json を更新する")
    args = ap.parse_args()

    forum = json.loads(FORUM_PATH.read_text(encoding="utf-8"))
    valid = known_slugs()
    parsed = parse_answers()

    pending = {
        tid for tid, v in forum.items()
        if v.get("review_status") == "pending" and v.get("answerable_standalone")
    }

    adopt, exclude, held, unknown_slug = {}, {}, {}, {}

    for tid, r in parsed.items():
        if tid not in pending:
            continue
        if r["verdict"] == "EXCLUDE":
            # 「読めなかった」起因のEXCLUDEは判定として無効 → 保留
            if UNREADABLE_PAT.search(r["reason"]):
                held[tid] = r
            else:
                exclude[tid] = r
            continue
        bad = [s for s in r["slugs"] if s not in valid]
        good = [s for s in r["slugs"] if s in valid]
        if bad:
            unknown_slug[tid] = (bad, good, r)
        if not good:
            held[tid] = r  # 採用できるslugが無い → 保留
            continue
        adopt[tid] = {**r, "slugs": good}

    missed = sorted(pending - set(parsed))

    print(f"pending {len(pending)} / 表から拾えた {len(set(parsed) & pending)}")
    print(f"  ADOPT   : {len(adopt)}")
    print(f"  EXCLUDE : {len(exclude)}")
    print(f"  保留(要人手): {len(held)}  ← 「読めなかった」EXCLUDE / 有効slug無し")
    print(f"  出力に無い : {len(missed)} {missed if missed else ''}")
    if unknown_slug:
        print(f"\n  ⚠️ 実在しないslugを含む判定 {len(unknown_slug)} 件（該当slugのみ除外して採用）:")
        for tid, (bad, good, r) in unknown_slug.items():
            print(f"    {tid}: 不明={bad} / 採用={good or '無し→保留'}")

    if not args.apply:
        print("\n(dry-run。--apply で forum_review.json を更新)")
        return

    for tid, r in adopt.items():
        v = forum[tid]
        v["expected_source"] = r["slugs"]
        v["match_type"] = normalize_mt(r["match_type"], r["slugs"])
        v["answerable_standalone"] = True
        v["exclusion_reason"] = None
        v["review_status"] = "reviewed"
        v["source_confirmation"] = "notebooklm"
        v["flags"] = []
        v["review_note"] = f"NotebookLM({r['src']}): {r['reason'][:120]}"

    for tid, r in exclude.items():
        v = forum[tid]
        v["expected_source"] = []
        v["match_type"] = None
        v["answerable_standalone"] = False
        v["exclusion_reason"] = r["reason"][:200]
        v["review_status"] = "reviewed"
        v["source_confirmation"] = "notebooklm"
        v["flags"] = []

    FORUM_PATH.write_text(
        json.dumps(forum, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n更新: ADOPT {len(adopt)} / EXCLUDE {len(exclude)} → {FORUM_PATH}")
    print(f"保留 {len(held)} 件と未処理 {len(missed)} 件は pending のまま（要人手・再投入）")


if __name__ == "__main__":
    main()
