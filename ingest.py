"""SoundQuest 記事をスクレイプ・チャンク化する処理モジュール。

このファイルは「純粋な処理関数」だけを持つ。
- DB への投入（upsert）は retriever.py（Qdrant版）が担当する。
- 起動（CLI / イベント）は main.py の Inngest function が担当する。

main.py からはこの2つの関数を step として呼ぶ:
    entries = scrape(url)                 -> list[dict]
    chunks  = chunk(entries, source_id)   -> list[dict]
"""
from __future__ import annotations

import json
import re
import time

import requests
from bs4 import BeautifulSoup

import config
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ══════════════════════════════════════════════════════════════
#  scrape: SoundQuest 記事 → entries（種別付きテキストのリスト）
#  main.py の step "scrape" から呼ばれる
# ══════════════════════════════════════════════════════════════

def scrape(url: str) -> list[dict]:
    """
    SoundQuest の1記事（複数ページ対応）をスクレイプする。
    返り値の各エントリ:
      {"text": str, "type": "text"|"audio"|"image", "source_url": str}
    """
    entries: list[dict] = []
    current_url: str | None = url
    visited: set[str] = set()

    while current_url:
        if current_url in visited:
            break
        visited.add(current_url)
        print(f"    fetch: {current_url}")

        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(current_url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"    retry {attempt + 1}/3: {e}")
                time.sleep(1.5)
        if resp is None:
            print(f"    warn: 取得失敗のためスキップ ({current_url})")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        content = soup.select_one("div.post-content")
        if not content:
            print(f"    warn: .post-content が見つかりません ({current_url})")
            break

        # ノイズ除去
        for tag in content.select(
            "script, style, nav, .ez-toc-container, .post-tags, "
            ".post-nav-links, .easy-footnote"
        ):
            tag.decompose()

        entries += _parse_content_blocks(content, current_url)

        current_url = _find_next_page(soup, current_url)
        if current_url:
            time.sleep(0.5)  # サーバー負荷軽減

    return entries


def _parse_content_blocks(content, page_url: str) -> list[dict]:
    """本文 DOM を走査し、テキスト・音声・画像を種別付きで抽出する。"""
    entries: list[dict] = []

    for elem in content.children:
        if not hasattr(elem, "name") or not elem.name:
            continue

        # ── テキストブロック ──
        if elem.name in {"p", "h2", "h3", "h4", "h5", "dl", "blockquote", "ul", "ol"}:
            text = _clean_text(elem.get_text(separator=" "))
            if text:
                entries.append({"text": text, "type": "text", "source_url": page_url})

        # ── .image ブロック（画像 + 直後の音声プレーヤーを一体で処理）──
        elif "image" in elem.get("class", []):
            block_text = _extract_image_block(elem)
            if block_text:
                entries.append({"text": block_text, "type": "image", "source_url": page_url})

        # ── 単体の audio ──
        elif elem.name == "audio" or elem.select("audio"):
            audio_text = _extract_audio_text(elem, surrounding="")
            if audio_text:
                entries.append({"text": audio_text, "type": "audio", "source_url": page_url})

        # ── Spotify 埋め込み ──
        elif elem.name == "div" and "sp-playlist" in elem.get("class", []):
            iframe = elem.select_one("iframe[src*='open.spotify.com']")
            if iframe:
                src = iframe.get("src", "")
                entries.append({
                    "text": f"[Spotify プレイリスト埋め込み] URL: {src}",
                    "type": "text",
                    "source_url": page_url,
                })

        # ── .def（用語定義ボックス）──
        elif elem.name == "dl" and "def" in elem.get("class", []):
            text = _clean_text(elem.get_text(separator=" "))
            if text:
                entries.append({"text": f"[用語定義] {text}", "type": "text", "source_url": page_url})

    return entries


def _extract_image_block(elem) -> str:
    """.image ブロックから画像 alt と直後の audio MP3 URL を合成する。"""
    parts: list[str] = []

    caption_el = elem.select_one(".imgcaption")
    if caption_el:
        cap = _clean_text(caption_el.get_text())
        if cap:
            parts.append(f"[図のキャプション] {cap}")

    for img in elem.select("img"):
        alt = img.get("alt", "").strip()
        src = img.get("src", "")
        if alt:
            parts.append(f"[画像の説明] {alt}")
        if src:
            parts.append(f"[画像URL] {src}")

    audio = elem.select_one("audio.wp-audio-shortcode")
    if audio:
        surrounding = " ".join(p.strip() for p in parts)
        parts.append(_extract_audio_text(audio, surrounding))

    return "\n".join(filter(None, parts))


def _extract_audio_text(audio_elem, surrounding: str) -> str:
    """audio 要素から MP3 URL を取り出し、周辺テキストと合成する。"""
    source = audio_elem.select_one("source[type='audio/mpeg']")
    if not source:
        return ""
    mp3_url = source.get("src", "").split("?")[0]
    if not mp3_url:
        return ""
    ctx = f" （周辺文脈: {surrounding[:120]}）" if surrounding else ""
    return f"[音声サンプル] MP3: {mp3_url}{ctx}"


def _page_num(url: str) -> int:
    """URL 末尾のページ番号を返す。番号が無ければ 1。"""
    m = re.search(r"/(\d+)/?$", url)
    return int(m.group(1)) if m else 1


def _find_next_page(soup: BeautifulSoup, current_url: str) -> str | None:
    """現在ページより大きい最小のページ番号の URL を返す。なければ None。"""
    nav = soup.select_one("p.post-nav-links")
    if not nav:
        return None
    cur = _page_num(current_url)
    candidates: list[tuple[int, str]] = []
    for a in nav.select("a.post-page-numbers"):
        href = a.get("href", "")
        if not href:
            continue
        n = _page_num(href)
        if n > cur:
            candidates.append((n, href))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _clean_text(text: str) -> str:
    """空白と制御文字を正規化する。"""
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════
#  chunk: entries → チャンク（メタ付き dict のリスト）
#  main.py の step "chunk" から呼ばれる
#
#  ※ MVP は既存の固定窓ロジックを流用（Phase 2 で構造ベースに刷新）。
#    entries を1本のテキストに結合してから分割する、従来の挙動を踏襲。
# ══════════════════════════════════════════════════════════════

def _looks_like_heading(text: str, entry_type: str = "text") -> bool:
    """見出しらしさのheuristic判定。DOMタグ情報が失われているための代替手段。"""
    if entry_type != "text":
        return False
    if re.match(r"^\d+\.\s", text):
        return True
    if len(text) < 30 and not text.endswith("。"):
        return True
    return False


def chunk_structure(entries: list[dict], source_id: str) -> list[dict]:
    """
    見出しheuristicでセクション分割し、長すぎれば再分割・短すぎればmergeする。
    chunk()と同じ入出力シグネチャ:
      返り値の各チャンク: {"text": str, "source": str, "chunk_index": int}
    追加で "heading"（breadcrumb文字列）をpayload用に含める。
    """
    sections: list[dict] = []
    current = {"heading": "", "texts": []}

    for e in entries:
        text = e.get("text", "")
        entry_type = e.get("type", "text")
        if not text:
            continue
        if _looks_like_heading(text, entry_type):
            if current["texts"] or current["heading"]:
                sections.append(current)
            current = {"heading": text, "texts": []}
        else:
            current["texts"].append(text)

    if current["texts"] or current["heading"]:
        sections.append(current)

    merged_sections: list[dict] = []
    pending_headings: list[str] = []
    pending_texts: list[str] = []

    for sec in sections:
        body = "\n\n".join(pending_texts + sec["texts"])
        headings = pending_headings + ([sec["heading"]] if sec["heading"] else [])

        if len(body) < config.MIN_SECTION_CHARS:
            pending_headings = headings
            pending_texts = pending_texts + sec["texts"]
            continue

        merged_sections.append({"breadcrumb": " > ".join(headings), "text": body})
        pending_headings = []
        pending_texts = []

    if pending_texts or pending_headings:
        if merged_sections:
            last = merged_sections[-1]
            if pending_headings:
                last["breadcrumb"] = last["breadcrumb"] + " > " + " > ".join(pending_headings)
            last["text"] = last["text"] + "\n\n" + "\n\n".join(pending_texts)
        else:
            merged_sections.append({
                "breadcrumb": " > ".join(pending_headings),
                "text": "\n\n".join(pending_texts),
            })

    chunks: list[dict] = []
    chunk_index = 0

    for sec in merged_sections:
        pieces = _chunk_text(sec["text"]) if len(sec["text"]) > config.CHUNK_CHARS else [sec["text"]]
        for piece in pieces:
            breadcrumb = sec["breadcrumb"]
            prefixed = f"[{breadcrumb}]\n{piece}" if breadcrumb else piece
            chunks.append({
                "text": prefixed,
                "source": source_id,
                "chunk_index": chunk_index,
                "heading": breadcrumb,
            })
            chunk_index += 1

    return chunks


def chunk(entries: list[dict], source_id: str) -> list[dict]:
    """
    entries を結合 → 固定窓で分割し、メタ付きチャンクのリストを返す。
    返り値の各チャンク:
      {"text": str, "source": str, "chunk_index": int}
    """
    full_text = "\n\n".join(e["text"] for e in entries if e.get("text"))
    pieces = _chunk_text(full_text)
    return [
        {"text": piece, "source": source_id, "chunk_index": i}
        for i, piece in enumerate(pieces)
    ]


def _chunk_text(text: str) -> list[str]:
    """改行を優先しつつ、文字数ベースでオーバーラップ付きに分割する。"""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    n = len(text)
    while start < n:
        end = min(start + config.CHUNK_CHARS, n)
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", "\n", "。", ". "):
                idx = window.rfind(sep)
                if idx > config.CHUNK_CHARS * 0.4:
                    end = start + idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = max(end - config.CHUNK_OVERLAP, end if end > start else start + 1)
    return chunks


def url_to_source_id(url: str) -> str:
    """URL から安定した source_id を生成する。"""
    return url.replace("https://", "").replace("http://", "").replace("/", "_").rstrip("_")

def load_raw(source_id: str) -> list[dict]:
    """data/raw/{source_id}.json を読んで entries を返す。"""
    path = RAW_DIR / f"{source_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))