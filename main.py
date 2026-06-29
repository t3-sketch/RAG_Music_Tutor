"""
music-rag のオーケストレーション層。
各 step の「入力 → 出力の型」をここで先に固定する。
実装は各モジュール（ingest / embedder / retriever / llm / audio）に委譲する。

このファイルの構造:
  ① import
  ② inngest_client 定義
  ③ rag_ingest function
  ④ rag_query function
  ⑤ app = FastAPI() / serve
"""
from __future__ import annotations

import logging

import inngest
import inngest.fast_api
import pydantic
from fastapi import FastAPI
from dotenv import load_dotenv

import config
from custom_types import (
    # ingest 側
    ScrapeEntry,
    ChunkWithSource,
    UpsertResult,
    # query 側
    QueryEventData,
    RetrievedChunk,
    QueryResult,
)

# 処理モジュール（接着剤である main.py だけが全部を知る）
import ingest
import embedder
import retriever
import audio
import llm

load_dotenv()


# ─────────────────────────────────────────────
#  ② Inngest client（function より先に定義する）
# ─────────────────────────────────────────────
inngest_client = inngest.Inngest(
    app_id="music-rag",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# ─────────────────────────────────────────────
#  ingest 側のイベント入力モデル
# ─────────────────────────────────────────────
class IngestEventData(pydantic.BaseModel):
    source_id: str


# =============================================================
#  rag_ingest function
#  流れ: load-raw → chunk → embed → upsert
# =============================================================
@inngest_client.create_function(
    fn_id="rag-ingest",
    trigger=inngest.TriggerEvent(event="rag/ingest"),
    concurrency=[inngest.Concurrency(limit=1)],
)
async def rag_ingest(ctx: inngest.Context) -> dict:
    data = IngestEventData.model_validate(ctx.event.data)
    source_id = data.source_id

    # ── step1: ローカル JSON 読み込み ───────────────
    # ingest.load_raw は list[dict] を返す。
    # output_type で step 境界で list[ScrapeEntry] に検証・変換する。
    # ここで type が "text"|"audio"|"image" 以外だったり
    # フィールドが欠けていれば ValidationError で早期に落ちる。
    entries: list[ScrapeEntry] = await ctx.step.run(
        "load-raw",
        lambda: ingest.load_raw(source_id),
        output_type=list[ScrapeEntry],
    )

    # ── step2: チャンク分割 ─────────────────────────
    # ingest.chunk は純粋関数で list[dict] を期待する
    # （custom_types を import しない設計）。
    # 接着剤である main.py で ScrapeEntry → dict に戻してから渡す。
    chunks: list[dict] = await ctx.step.run(
        "chunk",
        lambda: ingest.chunk([e.model_dump() for e in entries], source_id),
    )

    # ── step3: embed ───────────────────────────────
    # chunk のテキスト部分だけを embedder に渡す。
    vectors: list[list[float]] = await ctx.step.run(
        "embed",
        lambda: embedder.embed_documents([c["text"] for c in chunks]),
    )

    # ── step4: Qdrant upsert ───────────────────────
    # retriever.upsert は UpsertResult を返す。
    result: UpsertResult = await ctx.step.run(
        "upsert",
        lambda: retriever.upsert(chunks, vectors),
        output_type=UpsertResult,
    )

    return result.model_dump()


# =============================================================
#  rag_query function
#  流れ: embed → search →（任意）audio → generate
# =============================================================
@inngest_client.create_function(
    fn_id="rag-query",
    trigger=inngest.TriggerEvent(event="rag/query"),
)
async def rag_query(ctx: inngest.Context) -> dict:
    data = QueryEventData.model_validate(ctx.event.data)

    # ── step1: クエリを embed ───────────────────────
    query_vector: list[float] = await ctx.step.run(
        "embed-query",
        lambda: embedder.embed_query(data.query),
    )

    # ── step2: Qdrant 検索 ──────────────────────────
    # retriever.search は list[dict] を返す（{"text","source","score"}）。
    found: list[dict] = await ctx.step.run(
        "search",
        lambda: retriever.search(query_vector, data.top_k),
    )

    # ── step3: 音声解析（任意）─────────────────────
    # songle_url が None のときはスキップして None を返す。
    def _analyze_audio() -> str | None:
        if not data.songle_url:
            return None
        songle_data = audio.fetch_songle(data.songle_url)
        return audio.describe_songle(songle_data)

    audio_desc: str | None = await ctx.step.run(
        "analyze-audio",
        _analyze_audio,
    )

    # ── 変換（step の外。dict の組み替えはリトライ不要）──
    # retriever は {"text","source","score"} の dict を返すが、
    # llm.explain は c["meta"]["source"] / c["text"] を期待する。
    # この不一致をここ（接着剤）で吸収する。
    chunks_for_llm = [
        {"text": c["text"], "meta": {"source": c["source"]}}
        for c in found
    ]

    # ── step4: Gemini で生成 ────────────────────────
    answer: str = await ctx.step.run(
        "generate",
        lambda: llm.explain(data.query, chunks_for_llm, audio_desc),
    )

    return QueryResult(
        answer=answer,
        sources=[c["source"] for c in found],
        num_contexts=len(found),
    ).model_dump()


# ─────────────────────────────────────────────
#  ⑤ FastAPI 入口（薄い箱）+ Inngest マウント
# ─────────────────────────────────────────────
app = FastAPI()

inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest, rag_query],
)