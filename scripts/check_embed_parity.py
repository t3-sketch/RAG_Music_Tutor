"""local(FlagEmbedding) と nvidia(NVIDIA Build API) の bge-m3 埋め込みパリティ検証。

デプロイでクエリ埋め込みだけ nvidia バックエンドに切り替える前提
（corpus は local で埋め込み済み）なので、両者のベクトルが実用上一致する
ことがゲート条件になる。合格基準:
- 同一テキストの cosine 類似度 >= 0.99（NVIDIA側はfp16サービングのため厳密1.0は期待しない）
- Qdrant top-5 検索の (source, chunk_index) 重なり >= 4/5、かつ top-1 一致

実行（repoルートから。NVIDIA_API_KEY が必要）:
    uv run python scripts/check_embed_parity.py [collection]
collection 省略時は config.OPEN_COLLECTION（music_theory_open）。
"""
from __future__ import annotations

import math
import sys

from music_rag import config, embedder, retriever

# 埋め込みパリティ用テキスト（コーパスの語彙に近い日本語）
TEXTS = [
    "ドミナントセブンスコードはトニックへ解決する強い進行感を持つ",
    "短調の平行調は長調と同じ調号を共有する",
    "サブドミナントからトニックへの進行をプラガル終止と呼ぶ",
    "テンポとビートの関係について説明してください",
    "コード進行 ii-V-I はジャズで最も基本的なカデンツである",
]

# 検索パリティ用クエリ（eval質問に近い形）
QUERIES = [
    "トニックとドミナントの違いは何ですか",
    "セカンダリードミナントとは何ですか",
    "平行調と同主調の違いを教えてください",
    "終止（カデンツ）にはどんな種類がありますか",
    "モードとスケールの違いは何ですか",
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


def embed_both(texts: list[str]) -> tuple[list[list[float]], list[list[float]]]:
    """同じテキスト群を local / nvidia の両バックエンドで埋め込む。

    config.EMBED_BACKEND を一時的に差し替える（プロセス内のみ・検証専用）。
    """
    original = config.EMBED_BACKEND
    try:
        config.EMBED_BACKEND = "local"
        local_vecs = embedder.embed_documents(texts)
        config.EMBED_BACKEND = "nvidia"
        nvidia_vecs = embedder.embed_documents(texts)
    finally:
        config.EMBED_BACKEND = original
    return local_vecs, nvidia_vecs


def main() -> int:
    if not config.NVIDIA_API_KEY:
        print("NVIDIA_API_KEY が未設定です（.env に追加してください）")
        return 1

    collection = sys.argv[1] if len(sys.argv) > 1 else config.OPEN_COLLECTION
    ok = True

    print("== 1. ベクトルパリティ（cosine >= 0.99）==")
    local_vecs, nvidia_vecs = embed_both(TEXTS)
    for text, lv, nv in zip(TEXTS, local_vecs, nvidia_vecs):
        sim = cosine(lv, nv)
        mark = "OK " if sim >= 0.99 else "NG "
        if sim < 0.99:
            ok = False
        print(f"  {mark} cos={sim:.6f}  {text[:24]}...")

    print(f"\n== 2. 検索パリティ（{collection} top-5、重なり>=4/5・top-1一致）==")
    local_q, nvidia_q = embed_both(QUERIES)
    for query, lv, nv in zip(QUERIES, local_q, nvidia_q):
        hits_l = retriever.search(lv, top_k=5, collection=collection)
        hits_n = retriever.search(nv, top_k=5, collection=collection)
        keys_l = [(h["source"], h.get("chunk_index")) for h in hits_l]
        keys_n = [(h["source"], h.get("chunk_index")) for h in hits_n]
        overlap = len(set(keys_l) & set(keys_n))
        top1 = bool(keys_l) and keys_l[0] == keys_n[0]
        passed = overlap >= 4 and top1
        if not passed:
            ok = False
        mark = "OK " if passed else "NG "
        print(f"  {mark} overlap={overlap}/5 top1一致={top1}  {query[:20]}...")
        if not passed:
            print(f"       local : {keys_l}")
            print(f"       nvidia: {keys_n}")

    print("\n結果:", "PASS（nvidiaバックエンドをデプロイに使用可）" if ok
          else "FAIL（リモート再embedのコンティンジェンシーを検討）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
