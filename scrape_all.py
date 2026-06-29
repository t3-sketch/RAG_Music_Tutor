"""SoundQuest の全記事をスクレイプし、ローカルに JSON 保存するスクリプト。

役割（段1）:
  - クエストトップ（https://soundquest.jp/quest/）から記事URL一覧を収集
  - 各記事を ingest.scrape() で取得
  - 結果を data/raw/{source_id}.json に保存

設計方針:
  - SoundQuest にアクセスするのはこのスクリプトだけ。
    一度ローカルに落とせば、以降の chunk/embed/upsert 実験は
    data/raw/ から読むだけで完結する（Phase 2 の再ingestで再アクセス不要）。
  - サーバー負荷軽減のため、記事間に SLEEP_BETWEEN_ARTICLES 秒待つ。
  - 既に保存済みの記事はスキップする（再実行に安全＝冪等）。
    --force で強制再取得。

使い方:
    uv run python scrape_all.py            # 未取得の記事だけ取得
    uv run python scrape_all.py --force    # 全記事を取り直す
    uv run python scrape_all.py --limit 5  # 先頭5記事だけ（動作確認用）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import ingest

# ── 設定 ──────────────────────────────────────────────
QUEST_TOP = "https://soundquest.jp/quest/"
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

# 記事間の待機秒数（個人サイトへの礼儀。短くしすぎないこと）
SLEEP_BETWEEN_ARTICLES = 3.0


# ── 記事URL一覧の収集 ─────────────────────────────────
def collect_article_urls() -> list[str]:
    """クエストトップから記事URL一覧を集める。

    トップページのリンクは各記事の1ページ目を指す。
    ページ送り（.../2/ など）は ingest.scrape() が自動で辿るので
    ここには含めない（実際トップにも出てこない）。
    トップページ自身（QUEST_TOP）は記事ではないので除外する。
    """
    resp = requests.get(QUEST_TOP, headers=ingest.HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    urls: set[str] = set()
    for a in soup.select('a[href*="/quest/"]'):
        href = a.get("href", "")
        if not href.startswith("http"):
            continue
        # トップページ自身は除外
        if href.rstrip("/") == QUEST_TOP.rstrip("/"):
            continue
        urls.add(href)

    return sorted(urls)


# ── 保存 ──────────────────────────────────────────────
def save_entries(source_id: str, entries: list[dict]) -> Path:
    """entries を data/raw/{source_id}.json に保存する。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{source_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return path


def already_saved(source_id: str) -> bool:
    """既に保存済みかどうか。"""
    return (RAW_DIR / f"{source_id}.json").exists()


# ── メイン ────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="SoundQuest 全記事スクレイプ")
    parser.add_argument(
        "--force",
        action="store_true",
        help="保存済みの記事も取り直す",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="先頭 N 記事だけ取得（動作確認用）",
    )
    args = parser.parse_args()

    print("クエストトップから記事URLを収集中...")
    urls = collect_article_urls()
    if args.limit:
        urls = urls[: args.limit]
    print(f"対象記事数: {len(urls)}\n")

    saved, skipped, failed = 0, 0, 0

    for i, url in enumerate(urls, 1):
        source_id = ingest.url_to_source_id(url)

        # 既に保存済みならスキップ（--force 指定時を除く）
        if not args.force and already_saved(source_id):
            print(f"[{i}/{len(urls)}] skip（保存済み）: {source_id}")
            skipped += 1
            continue

        print(f"[{i}/{len(urls)}] scrape: {url}")
        try:
            entries = ingest.scrape(url)
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1
            # 失敗しても次へ進む。ウェイトは入れる。
            time.sleep(SLEEP_BETWEEN_ARTICLES)
            continue

        if not entries:
            print("    warn: entries が空（本文が取れなかった可能性）")
            failed += 1
            time.sleep(SLEEP_BETWEEN_ARTICLES)
            continue

        path = save_entries(source_id, entries)
        print(f"    saved: {path.name}（{len(entries)} entries）")
        saved += 1

        # サーバー負荷軽減（最後の記事の後は待たない）
        if i < len(urls):
            time.sleep(SLEEP_BETWEEN_ARTICLES)

    print("\n========== 完了 ==========")
    print(f"  保存: {saved}")
    print(f"  スキップ（保存済み）: {skipped}")
    print(f"  失敗: {failed}")
    print(f"  保存先: {RAW_DIR}")


if __name__ == "__main__":
    main()
