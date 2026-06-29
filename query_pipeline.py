"""同期版クエリパイプライン（Inngest を介さない接着剤）。
main.py の rag_query と同じ流れを、その場で結果を返す同期関数にしたもの。
Streamlit など同期UIから直接呼ぶ。"""
from __future__ import annotations

import config
import embedder
import retriever
import audio
import llm


def answer_query(query: str, top_k: int = config.TOP_K,
                 songle_url: str | None = None) -> dict:
    query_vector = embedder.embed_query(query)          # step1: embed
    found = retriever.search(query_vector, top_k)        # step2: search（既定で music_theory）

    audio_desc = None                                    # step3: 音声（任意）
    if songle_url:
        audio_desc = audio.describe_songle(audio.fetch_songle(songle_url))

    # retriever の {"text","source","score"} を llm.explain が期待する形に詰め替え
    chunks_for_llm = [{"text": c["text"], "meta": {"source": c["source"]}} for c in found]

    answer = llm.explain(query, chunks_for_llm, audio_desc)   # step4: generate
    return {"answer": answer,
            "sources": [c["source"] for c in found],
            "contexts": found}