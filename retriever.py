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
    search(query_vector, top_k) -> [{"text","source","score"}, ...]
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient, models

import config

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

    points = [
        models.PointStruct(
            id=_point_id(chunk["source"], chunk["chunk_index"]),
            vector=vector,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
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

    results: list[dict] = []
    for point in response.points:
        payload = point.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "score": float(point.score),
            }
        )
    return results
