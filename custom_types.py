"""step 境界（JSON シリアライズをまたぐ箇所）の型定義。

設計の原則:
- 各 Inngest step の入出力のうち「複数フィールドが意味的にまとまっているもの」を
  Pydantic モデルにする。プリミティブ単体（list[float] など）はモデルにしない。
- 処理モジュール（ingest.py / retriever.py / llm.py）はこれらの型を import しない。
  型に詰めるのは main.py（接着剤）の責務。

依存方向:
    custom_types  ←  main.py  →  ingest / retriever / llm
                        ↑ main.py だけが両方を知る
"""
from __future__ import annotations

import pydantic


# ══════════════════════════════════════════════════════════════
#  ingest 側（rag_ingest function）
# ══════════════════════════════════════════════════════════════

class ScrapeEntry(pydantic.BaseModel):
    """scrape step の出力1件。SoundQuest の1ブロックに対応。"""
    text: str
    type: str          # "text" | "audio" | "image"
    source_url: str


class ChunkWithSource(pydantic.BaseModel):
    """chunk step の出力1件。embed に渡す最小単位。"""
    text: str
    source: str
    chunk_index: int


class UpsertResult(pydantic.BaseModel):
    """upsert step の出力。投入したチャンク数。"""
    ingested: int
    source: str


# ══════════════════════════════════════════════════════════════
#  query 側（rag_query function）
# ══════════════════════════════════════════════════════════════

class QueryEventData(pydantic.BaseModel):
    """rag/query イベントの入力。"""
    query: str
    songle_url: str | None = None     # 任意。音声解析するときだけ
    top_k: int = 5


class RetrievedChunk(pydantic.BaseModel):
    """search step の出力1件。dashboard で確認できるよう score も持つ。"""
    text: str
    source: str
    score: float


class QueryResult(pydantic.BaseModel):
    """rag_query function の最終出力。"""
    answer: str
    sources: list[str]
    num_contexts: int