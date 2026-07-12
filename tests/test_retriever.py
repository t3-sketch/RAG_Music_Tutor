"""retriever の characterization テスト。

- _point_id: uuid5決定性は ingest 冪等性（再投入で points 不変）の根拠そのもの
- search: payload → 返り値dict のマッピング契約（chunk_index を含む。MLOps traceの前提）
- upsert: クライアントに触る前の入力検証

Qdrant 本体は使わない（fakeクライアントを注入）。ネットワーク・Docker不要で回る。
"""
import uuid
from types import SimpleNamespace

import pytest

from music_rag import retriever


class TestPointId:
    def test_deterministic(self):
        assert retriever._point_id("src", 0) == retriever._point_id("src", 0)

    def test_matches_uuid5_url_namespace(self):
        """ID仕様 = uuid5(NAMESPACE_URL, "{source}:{chunk_index}")。
        この式が変わると既存collection全point IDと不整合になり冪等性が壊れる。"""
        assert retriever._point_id("src", 3) == str(
            uuid.uuid5(uuid.NAMESPACE_URL, "src:3")
        )

    def test_distinct_by_source_and_index(self):
        ids = {
            retriever._point_id("a", 0),
            retriever._point_id("a", 1),
            retriever._point_id("b", 0),
        }
        assert len(ids) == 3


class _FakeClient:
    """search が使う2メソッドだけ持つ最小フェイク。"""

    def __init__(self, points):
        self._points = points

    def collection_exists(self, name):
        return True

    def query_points(self, collection_name, query, limit, with_payload):
        return SimpleNamespace(points=self._points[:limit])


class TestSearch:
    def test_payload_mapping_includes_chunk_index(self, monkeypatch):
        point = SimpleNamespace(
            score=0.87,
            payload={
                "text": "本文", "source": "art1", "chunk_index": 5,
                "heading": "H", "title": "T", "source_url": "u",
                "book": "B", "license": "CC", "license_url": "cu",
            },
        )
        monkeypatch.setattr(retriever, "_client", lambda: _FakeClient([point]))
        [hit] = retriever.search([0.0], top_k=1)
        assert hit == {
            "text": "本文", "source": "art1", "chunk_index": 5, "score": 0.87,
            "heading": "H", "title": "T", "source_url": "u",
            "book": "B", "license": "CC", "license_url": "cu",
        }

    def test_old_points_without_meta_fall_back_to_none(self, monkeypatch):
        """旧collection（出典メタ導入前）のpointでも落ちずNoneで返す契約。"""
        point = SimpleNamespace(score=0.5, payload={"text": "t", "source": "s"})
        monkeypatch.setattr(retriever, "_client", lambda: _FakeClient([point]))
        [hit] = retriever.search([0.0], top_k=1)
        assert hit["chunk_index"] is None
        assert hit["heading"] is None

    def test_missing_collection_returns_empty(self, monkeypatch):
        fake = _FakeClient([])
        fake.collection_exists = lambda name: False
        monkeypatch.setattr(retriever, "_client", lambda: fake)
        assert retriever.search([0.0]) == []


class TestUpsertValidation:
    def test_empty_chunks_short_circuit(self):
        assert retriever.upsert([], []) == {"ingested": 0, "source": ""}

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            retriever.upsert(
                [{"text": "t", "source": "s", "chunk_index": 0}],
                [[0.0], [0.0]],
            )
