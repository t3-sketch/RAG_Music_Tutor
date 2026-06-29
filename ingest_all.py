"""data/raw/ の clean な記事だけを rag/ingest イベントとして送る起動側（段2）。

役割:
  - data/raw/*.json から source_id を列挙
  - gated_report.txt の "=== gated ===" 以降に載る gated 記事を除外
  - 残り（clean な162件）を rag/ingest イベントとして fan-out で send

scrape_all.py の兄弟（あちらは scrape の起動側、こちらは ingest の起動側）。
send するだけ。embed は rag_ingest（main.py）の step の中で起きる。

前提:
  - 先に Qdrant / FastAPI / Inngest dev server を起動しておくこと。
    docker-compose up -d
    uv run uvicorn main:app --reload
    npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest

使い方:
    uv run python ingest_all.py            # clean な全件を send
    uv run python ingest_all.py --limit 3  # 先頭3件だけ（動作確認用）
    uv run python ingest_all.py --dry-run  # send せず対象一覧だけ表示
"""
from __future__ import annotations

import argparse
from pathlib import Path

import inngest

from main import inngest_client

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
GATED_REPORT = Path(__file__).resolve().parent / "gated_report.txt"


def load_gated_ids() -> set[str]:
    """gated_report.txt の "=== gated ===" 以降から gated な source_id 集合を作る。

    各行は "<source_id>.json"。.json を外して source_id にする。
    レポートが無い / セクションが無い場合は空集合を返す（＝除外なし）が、警告する。
    """
    if not GATED_REPORT.exists():
        print(f"  warn: {GATED_REPORT.name} が見つかりません。除外なしで続行します。")
        return set()

    lines = GATED_REPORT.read_text(encoding="utf-8").splitlines()
    try:
        marker = next(i for i, ln in enumerate(lines) if ln.strip() == "=== gated ===")
    except StopIteration:
        print("  warn: '=== gated ===' セクションが見つかりません。除外なしで続行します。")
        return set()

    gated: set[str] = set()
    for ln in lines[marker + 1:]:
        name = ln.strip()
        if name:
            gated.add(name.removesuffix(".json"))
    return gated


def collect_clean_source_ids() -> list[str]:
    """data/raw/ の全 source_id から gated を除いた clean な一覧を返す。"""
    all_ids = sorted(p.stem for p in RAW_DIR.glob("*.json"))
    gated = load_gated_ids()
    clean = [sid for sid in all_ids if sid not in gated]

    print(f"  data/raw/ 合計: {len(all_ids)} 件")
    print(f"  gated（除外）: {len(gated)} 件")
    print(f"  clean（対象）: {len(clean)} 件\n")
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="clean な記事を rag/ingest で fan-out send")
    parser.add_argument("--limit", type=int, default=None, help="先頭 N 件だけ送る（動作確認用）")
    parser.add_argument("--dry-run", action="store_true", help="send せず対象一覧だけ表示")
    args = parser.parse_args()

    source_ids = collect_clean_source_ids()
    if args.limit:
        source_ids = source_ids[: args.limit]
        print(f"  --limit {args.limit}: 先頭 {len(source_ids)} 件に絞ります\n")

    if args.dry_run:
        for sid in source_ids:
            print(f"    [dry-run] {sid}")
        print(f"\n  dry-run: {len(source_ids)} 件（send しませんでした）")
        return

    # ── fan-out: 1 source_id = 1 event を一括 send ──
    # event id に source_id を使う → 二重実行しても同じ記事は1回しかトリガーされない（冪等）。
    events = [
        inngest.Event(
            name="rag/ingest",
            data={"source_id": sid},
            id=f"ingest:{sid}",
        )
        for sid in source_ids
    ]
    ids = inngest_client.send_sync(events)

    print(f"  send 完了: {len(events)} 件の rag/ingest イベントを送信しました。")
    print(f"  返却 event id 数: {len(ids)}")
    print("  進捗は Inngest dashboard（http://127.0.0.1:8288）で確認してください。")


if __name__ == "__main__":
    main()