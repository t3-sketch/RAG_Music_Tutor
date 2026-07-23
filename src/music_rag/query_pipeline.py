"""同期版クエリパイプライン（Inngest を介さない接着剤）。
main.py の rag_query と同じ流れを、その場で結果を返す同期関数にしたもの。
Streamlit など同期UIから直接呼ぶ。"""
from __future__ import annotations

from music_rag import config
from music_rag import embedder
from music_rag import retriever
from music_rag import llm


def answer_query(query: str, top_k: int = config.TOP_K,
                 audio_path: str | None = None) -> dict:
    # step1: query expansion（条件C。ENABLE_QE=false で素通し。失敗時も query が返る）
    search_query = llm.expand_query(query)

    # step2: embed → search。ハイブリッド（dense+sparse を RRF 融合）が既定。
    # ENABLE_HYBRID=false で従来の dense-only 経路へ切り戻す。
    if config.ENABLE_HYBRID:
        dense, sparse = embedder.embed_query_hybrid(search_query)
        found = retriever.search_hybrid(dense, sparse, top_k)
    else:
        found = retriever.search(embedder.embed_query(search_query), top_k)

    audio_desc = None                                    # step3: 音声（任意。ローカルパス or URL）
    if audio_path:
        # librosa 等の重い依存をテキスト経路に持ち込まないため、ここで import する
        # （テキストのみの軽量デプロイでは audio/librosa 自体がインストールされない）
        from music_rag import audio, audio_source

        local_path = audio_source.resolve(audio_path)    # URL なら一時ファイルに解決
        try:
            audio_desc = audio.describe(audio.analyze(local_path))
        finally:
            audio_source.cleanup(local_path, audio_path)

    # retriever の {"text","source","score"} を llm.explain が期待する形に詰め替え
    chunks_for_llm = [{"text": c["text"], "meta": {"source": c["source"]}} for c in found]

    answer = llm.explain(query, chunks_for_llm, audio_desc)   # step4: generate
    return {"answer": answer,
            "sources": [c["source"] for c in found],
            "contexts": found,
            "audio_desc": audio_desc}