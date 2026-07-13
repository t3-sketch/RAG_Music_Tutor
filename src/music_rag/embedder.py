"""BGE-M3 による埋め込み生成モジュール（純粋な処理関数のみ）。

責務は embedding だけ（Single Responsibility）。
- Inngest / custom_types は import しない。
- numpy を境界に出さない。返り値はすべて list[float] / list[list[float]]
  （JSON シリアライズ境界＝Inngest step をまたぐため）。

バックエンドは config.EMBED_BACKEND で切替:
- local     : FlagEmbedding（既定。ingest と同一経路。torch を使う）
- deepinfra : DeepInfra がホストする同一モデル BAAI/bge-m3 の OpenAI互換API。
              torch 不要の軽量デプロイ用。ベクトル空間は local と共有
              （同一モデルのため。パリティは scripts/check_embed_parity.py で検証）。
- nvidia    : NVIDIA Build の同一モデル（現在 bge-m3 が500で使用不可。経路のみ保持）。

deepinfra / nvidia はどちらも OpenAI互換の /embeddings なので共通経路で扱う。

main.py が期待するインターフェース:
    embed_query(text: str) -> list[float]                  # 1024 次元
    embed_documents(texts: list[str]) -> list[list[float]]
"""
from __future__ import annotations

from functools import lru_cache

from music_rag import config

# リモートAPIの1リクエストあたり入力件数。上限が非公開のため控えめに固定する。
_REMOTE_BATCH_SIZE = 32


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


@lru_cache(maxsize=4)
def _remote_client(base_url: str, api_key: str):
    """OpenAI互換クライアント。base_url/key ごとにキャッシュ（backend切替に安全）。"""
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def _remote_params() -> tuple[str, str, str, dict]:
    """現在のバックエンドの (base_url, api_key, model, extra_body) を返す。"""
    if config.EMBED_BACKEND == "deepinfra":
        return (config.DEEPINFRA_BASE_URL, config.DEEPINFRA_API_KEY,
                config.DEEPINFRA_EMBED_MODEL, {})
    # nvidia（bge-m3 は現在500。復旧時用に経路のみ保持）。
    # truncate=END で 8192 token 超の入力だけ安全に切る。
    return (config.NVIDIA_BASE_URL, config.NVIDIA_API_KEY,
            config.NVIDIA_EMBED_MODEL, {"truncate": "END"})


def _encode_dense_local(texts: list[str]) -> list[list[float]]:
    output = _model().encode(
        texts,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = output["dense_vecs"]
    return [[float(x) for x in row] for row in dense]


def _encode_dense_remote(texts: list[str]) -> list[list[float]]:
    """OpenAI互換の /embeddings（DeepInfra / NVIDIA）で埋め込む。入力順を保って返す。

    bge-m3 は instruction 不要の対称モデルなので input_type は付けない。
    """
    base_url, api_key, model, extra_body = _remote_params()
    client = _remote_client(base_url, api_key)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _REMOTE_BATCH_SIZE):
        batch = texts[start:start + _REMOTE_BATCH_SIZE]
        resp = client.embeddings.create(
            model=model,
            input=batch,
            extra_body=extra_body,
        )
        # API は index 付きで返すため、入力順に並べ直す
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors.extend([float(x) for x in d.embedding] for d in ordered)
    return vectors


def _encode_dense(texts: list[str]) -> list[list[float]]:
    if config.EMBED_BACKEND in ("deepinfra", "nvidia"):
        return _encode_dense_remote(texts)
    return _encode_dense_local(texts)


def embed_query(text: str) -> list[float]:
    """検索クエリ1件を 1024 次元の dense ベクトルへ変換する。"""
    return _encode_dense([text])[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """ドキュメント群を 1024 次元の dense ベクトル列へ変換する。"""
    if not texts:
        return []
    return _encode_dense(texts)
