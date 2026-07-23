"""
Qdrant を使ったベクトル検索のラッパー（純粋な処理関数のみ）。

責務は DB 操作だけ（Single Responsibility）。
- Inngest / custom_types は import しない。embedding もここではしない
  （embedding は embedder.py、型詰めは main.py の責務）。
- 返り値は素の dict。main.py 側で UpsertResult / RetrievedChunk に詰める。

main.py が期待するインターフェース:
    upsert(chunks, vectors) -> {"ingested": int, "source": str}
        chunks:  [{"text","source","chunk_index"}, ...]（ingest.chunk の出力）
        vectors: [[float, ...], ...]（embedder.embed_documents の出力, 1024次元）
    search(query_vector, top_k) -> [{"text","source","chunk_index","score"}, ...]
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

from music_rag import config

# source + chunk_index から安定 ID を作るための名前空間。
# 同じ source の再投入で同じ ID になり、重複せず上書きされる（冪等）。
_ID_NAMESPACE = uuid.NAMESPACE_URL

_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    """Qdrant クライアントを生成する（プロセス内で再利用）。"""
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def _ensure_collection(collection: str | None = None) -> None:
    """
    collection が無ければ作る、あれば再利用する（冪等）。
    collection 未指定なら config.QDRANT_COLLECTION を使う。
    """
    coll = collection or config.QDRANT_COLLECTION
    client = _client()
    if client.collection_exists(coll):
        return
    client.create_collection(
        collection_name=coll,
        vectors_config=models.VectorParams(
            size=config.EMBED_DIM,
            distance=_DISTANCE_MAP[config.DISTANCE],
        ),
    )


def _point_id(source: str, chunk_index: int) -> str:
    """source + chunk_index から決定的な UUID を生成する（再投入で冪等）。"""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{source}:{chunk_index}"))


def upsert(
    chunks: list[dict],
    vectors: list[list[float]],
    collection: str | None = None,
) -> dict:
    if not chunks:
        return {"ingested": 0, "source": ""}
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks と vectors の件数が一致しません: "
            f"{len(chunks)} != {len(vectors)}"
        )

    _ensure_collection(collection)        # ← 引数を渡す
    coll = collection or config.QDRANT_COLLECTION   # ← 追加

    # 必須3キーに加え、チャンクが持つ追加メタ（heading / source_url / title /
    # book / license 等の出典情報）もそのままpayloadへ通す。
    # SoundQuest系チャンクは追加キーを持たないため従来とpayload同一。
    points = [
        models.PointStruct(
            id=_point_id(chunk["source"], chunk["chunk_index"]),
            vector=vector,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                **{k: v for k, v in chunk.items()
                   if k not in ("text", "source", "chunk_index") and v is not None},
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    _client().upsert(collection_name=coll, points=points)

    source = chunks[0]["source"]
    return {"ingested": len(points), "source": source}


def search(
    query_vector: list[float],
    top_k: int = config.TOP_K,
    collection: str | None = None,
) -> list[dict]:
    coll = collection or config.QDRANT_COLLECTION   # ← 追加
    client = _client()
    if not client.collection_exists(coll):          # ← coll に変更
        return []

    response = client.query_points(
        collection_name=coll,                       # ← coll に変更
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [_to_result(p) for p in response.points]


def _to_result(point) -> dict:
    """Qdrant の ScoredPoint を main.py / UI が期待する素の dict にする。"""
    payload = point.payload or {}
    return {
        "text": payload.get("text", ""),
        "source": payload.get("source", ""),
        # チャンク単位の追跡用（MLOps trace / eval のドリルダウンで
        # 「記事のどこが引かれたか」を特定する。upsert時から全payloadに存在）
        "chunk_index": payload.get("chunk_index"),
        "score": float(point.score),
        # 出典表示用メタ（旧ポイントには無いのでNone → UI側でフォールバック）
        "heading": payload.get("heading"),
        "title": payload.get("title"),
        "source_url": payload.get("source_url"),
        "book": payload.get("book"),
        "license": payload.get("license"),
        "license_url": payload.get("license_url"),
    }


def search_hybrid(
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    top_k: int = config.TOP_K,
    collection: str | None = None,
) -> list[dict]:
    """dense と sparse を別々に引いて RRF で融合する（named vectors 前提）。

    dense-only の `search` とは collection もベクトル構成も別なので、経路を分けて
    共存させる（ENABLE_HYBRID=false で従来経路へ切り戻せる）。
    """
    coll = collection or config.HYBRID_COLLECTION
    client = _client()
    if not client.collection_exists(coll):
        # dense 版の search() と違い、ここは空を返さず落とす。
        # ハイブリッド経路は QDRANT_COLLECTION を見ない（HYBRID_COLLECTION が別系統）ため、
        # 取り違えると「別コーパスを配信する / 無言で0件になる」まで気づけない。
        # 設定ミスは検索結果ではなく例外で表に出す。
        raise RuntimeError(
            f"ハイブリッド検索の collection '{coll}' が {config.QDRANT_URL} に無い。"
            f" HYBRID_COLLECTION を設定するか、ENABLE_HYBRID=false で dense 経路に切り戻す"
            f"（dense 側の切替は QDRANT_COLLECTION）。"
        )

    response = client.query_points(
        collection_name=coll,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=config.HYBRID_PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(sparse_vector.keys()),
                    values=list(sparse_vector.values()),
                ),
                using="sparse",
                limit=config.HYBRID_PREFETCH_LIMIT,
            ),
        ],
        # RRF: スコア尺度の違う dense/sparse を順位だけで融合する（正規化不要）
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [_to_result(p) for p in response.points]
