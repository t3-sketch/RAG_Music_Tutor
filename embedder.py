"""BGE-M3 による埋め込み生成モジュール（純粋な処理関数のみ）。

責務は embedding だけ（Single Responsibility）。
- Inngest / custom_types は import しない。
- numpy を境界に出さない。返り値はすべて list[float] / list[list[float]]
  （JSON シリアライズ境界＝Inngest step をまたぐため）。

main.py が期待するインターフェース:
    embed_query(text: str) -> list[float]                  # 1024 次元
    embed_documents(texts: list[str]) -> list[list[float]]
"""
from __future__ import annotations

from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel

import config


@lru_cache(maxsize=1)
def _model() -> BGEM3FlagModel:
    """BGE-M3 モデルを遅延ロードする（初回呼び出し時に約2GBをDL）。

    use_fp16 は CPU/MPS 実行での不安定さを避けるため無効にする
    （dense ベクトルの数値安定性を優先）。
    """
    return BGEM3FlagModel(config.EMBED_MODEL, use_fp16=False)


def _encode_dense(texts: list[str]) -> list[list[float]]:
    output = _model().encode(
        texts,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense = output["dense_vecs"]
    return [[float(x) for x in row] for row in dense]


def embed_query(text: str) -> list[float]:
    """検索クエリ1件を 1024 次元の dense ベクトルへ変換する。"""
    return _encode_dense([text])[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """ドキュメント群を 1024 次元の dense ベクトル列へ変換する。"""
    if not texts:
        return []
    return _encode_dense(texts)
