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
    # step1: 音声解析（任意。ローカルパス or URL）。search より前に置く。
    # 解析結果を検索クエリにも使うため（生成層プロンプトだけに渡していた頃は、
    # 「楽曲の解説をして」という中身の無い文字列だけで検索していた）。
    audio_desc = audio_terms = None
    if audio_path:
        # librosa 等の重い依存をテキスト経路に持ち込まないため、ここで import する
        # （テキストのみの軽量デプロイでは audio/librosa 自体がインストールされない）
        from music_rag import audio, audio_source

        local_path = audio_source.resolve(audio_path)    # URL なら一時ファイルに解決
        try:
            analysis = audio.analyze(local_path)
        finally:
            audio_source.cleanup(local_path, audio_path)
        audio_desc = audio.describe(analysis)            # 生成層向け（時系列を含む）
        audio_terms = audio.search_terms(analysis)       # 検索向け（キー＋主要コード）

    # step2: 検索クエリの組み立て。
    # 音声つきは解析結果を足す。QE は通さない ──「楽曲の解説をして」に一般語
    # （音楽理論 楽曲分析 …）を足すと総論記事が上位を占めて逆効果になるため。
    # テキストのみは従来どおり QE（条件C。ENABLE_QE=false で素通し、失敗時も query が返る）。
    search_query = f"{query}\n{audio_terms}" if audio_terms else llm.expand_query(query)

    # step3: embed → search。ハイブリッド（dense+sparse を RRF 融合）が既定。
    # ENABLE_HYBRID=false で従来の dense-only 経路へ切り戻す。
    if config.ENABLE_HYBRID:
        dense, sparse = embedder.embed_query_hybrid(search_query)
        found = retriever.search_hybrid(dense, sparse, top_k)
    else:
        found = retriever.search(embedder.embed_query(search_query), top_k)

    # retriever の {"text","source","score"} を llm.explain が期待する形に詰め替え
    chunks_for_llm = [{"text": c["text"], "meta": {"source": c["source"]}} for c in found]

    answer = llm.explain(query, chunks_for_llm, audio_desc)   # step4: generate
    return {"answer": answer,
            "sources": [c["source"] for c in found],
            "contexts": found,
            "audio_desc": audio_desc}