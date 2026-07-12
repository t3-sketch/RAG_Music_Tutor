"""ローカルQdrantのcollectionをQdrant Cloudへコピーする（vectors + payload + point ID保持）。

snapshot restoreではなくscroll→upsertを使う（バージョン非依存・数千pointなら数秒）。
再実行は冪等（point ID保持のためupsertが上書きになる）。

実行（repoルートから）:
    uv run python scripts/copy_collection.py \
        --dest-url https://xxx.cloud.qdrant.io:6333 \
        --dest-api-key <key> \
        [--collection music_theory_open] [--src-url http://localhost:6333]

--dest-api-key は環境変数 QDRANT_CLOUD_API_KEY でも指定可。
"""
from __future__ import annotations

import argparse
import os

from qdrant_client import QdrantClient, models

from music_rag import config

BATCH = 256


def copy_collection(src: QdrantClient, dest: QdrantClient, collection: str) -> int:
    if not dest.collection_exists(collection):
        dest.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=config.EMBED_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"created collection: {collection} ({config.EMBED_DIM}dim / cosine)")

    copied = 0
    offset = None
    while True:
        points, offset = src.scroll(
            collection_name=collection,
            limit=BATCH,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not points:
            break
        dest.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                for p in points
            ],
        )
        copied += len(points)
        print(f"  copied {copied} points...")
        if offset is None:
            break
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=config.OPEN_COLLECTION)
    parser.add_argument("--src-url", default="http://localhost:6333")
    parser.add_argument("--dest-url", required=True)
    parser.add_argument("--dest-api-key",
                        default=os.getenv("QDRANT_CLOUD_API_KEY"))
    args = parser.parse_args()

    src = QdrantClient(url=args.src_url)
    dest = QdrantClient(url=args.dest_url, api_key=args.dest_api_key)

    src_count = src.count(args.collection).count
    print(f"source: {args.src_url} / {args.collection} = {src_count} points")

    copied = copy_collection(src, dest, args.collection)
    dest_count = dest.count(args.collection).count
    print(f"dest:   {args.dest_url} / {args.collection} = {dest_count} points")

    if dest_count != src_count:
        print(f"NG: point数不一致（src {src_count} / dest {dest_count}）")
        return 1
    print(f"OK: {copied} points コピー完了・件数一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
