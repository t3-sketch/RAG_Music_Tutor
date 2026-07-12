"""BGE-M3 による埋め込み生成モジュール（純粋な処理関数のみ）。

責務は embedding だけ（Single Responsibility）。
- Inngest / custom_types は import しない。
- numpy を境界に出さない。返り値はすべて list[float] / list[list[float]]
  （JSON シリアライズ境界＝Inngest step をまたぐため）。

バックエンドは config.EMBED_BACKEND で切替:
- local  : FlagEmbedding（既定。ingest と同一経路。torch を使う）
- nvidia : NVIDIA Build がホストする同一モデル baai/bge-m3 の OpenAI互換API。
           torch 不要の軽量デプロイ用。ベクトル空間は local と共有
           （同一モデルのため。パリティは scripts/check_embed_parity.py で検証）。

main.py が期待するインターフェース:
    embed_query(text: str) -> list[float]                  # 1024 次元
    embed_documents(texts: list[str]) -> list[list[float]]
"""
from __future__ import annotations

from functools import lru_cache

from music_rag import config

# NVIDIA API の1リクエストあたり入力件数。上限が非公開のため控えめに固定する。
_NVIDIA_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def _model():
    """BGE-M3 モデルを遅延ロードする（初回呼び出し時に約2GBをDL）。

    FlagEmbedding の import もここで行う（モジュールロード時に torch を
    引き込まないため。nvidia バックエンドのデプロイでは torch 自体が無い）。
    use_fp16 は CPU/MPS 実行での不安定さを避けるため無効にする
    （dense ベクトルの数値安定性を優先）。
    """
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(config.EMBED_MODEL, use_fp16=False)


@lru_cache(maxsize=1)
def _nvidia_client():
    from openai import OpenAI

    return OpenAI(api_key=config.NVIDIA_API_KEY, base_url=config.NVIDIA_BASE_URL)


def _encode_dense_local(texts: list[str]) -> list[list[float]]:
    output = _model().encode(
        texts,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = output["dense_vecs"]
    return [[float(x) for x in row] for row in dense]


def _encode_dense_nvidia(texts: list[str]) -> list[list[float]]:
    """NVIDIA Build の /v1/embeddings で埋め込む。入力順を保って返す。

    bge-m3 の NIM スキーマに input_type パラメータは無い（instruction不要の
    対称モデルのため）。truncate=END で 8192 token 超の入力だけ安全に切る。
    """
    client = _nvidia_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _NVIDIA_BATCH_SIZE):
        batch = texts[start:start + _NVIDIA_BATCH_SIZE]
        resp = client.embeddings.create(
            model=config.NVIDIA_EMBED_MODEL,
            input=batch,
            extra_body={"truncate": "END"},
        )
        # API は index 付きで返すため、入力順に並べ直す
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors.extend([float(x) for x in d.embedding] for d in ordered)
    return vectors


def _encode_dense(texts: list[str]) -> list[list[float]]:
    if config.EMBED_BACKEND == "nvidia":
        return _encode_dense_nvidia(texts)
    return _encode_dense_local(texts)


def embed_query(text: str) -> list[float]:
    """検索クエリ1件を 1024 次元の dense ベクトルへ変換する。"""
    return _encode_dense([text])[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """ドキュメント群を 1024 次元の dense ベクトル列へ変換する。"""
    if not texts:
        return []
    return _encode_dense(texts)
