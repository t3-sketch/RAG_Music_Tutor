"""chunk_structure の characterization テスト（現挙動の固定）。

狙い:
- OMT対応で入れた「明示heading尊重・heuristic無効化」の契約（commit 09fffd9）を回帰から守る
- SoundQuest経路（heading型なし→heuristic）の従来挙動を固定する
- chunk_index の連番性と breadcrumb prefix は Qdrant point ID / 出典表示の前提
"""
from music_rag import config
from music_rag.ingest import _looks_like_heading, chunk_structure

# MIN_SECTION_CHARS(=100) を確実に超える本文
LONG = "これは本文です。" * 20  # 160字


def _entries(*pairs: tuple[str, str]) -> list[dict]:
    return [{"type": t, "text": x} for t, x in pairs]


class TestExplicitHeadings:
    """OMT経路: heading型entryがあれば、それだけがセクション境界になる。"""

    def test_headings_define_sections(self):
        chunks = chunk_structure(
            _entries(("heading", "Intro"), ("text", LONG),
                     ("heading", "Section A"), ("text", LONG)),
            "src",
        )
        assert [c["heading"] for c in chunks] == ["Intro", "Section A"]
        assert chunks[0]["text"].startswith("[Intro]\n")
        assert chunks[1]["text"].startswith("[Section A]\n")

    def test_heuristic_disabled_when_explicit_headings_exist(self):
        """短い見出しっぽいtext行が、明示heading存在下では境界にならない
        （修正前はこれが誤爆して 'Chapter Playlist' 等のゴミbreadcrumbを生んだ）。"""
        looks_like_heading = "Chapter Playlist"
        assert _looks_like_heading(looks_like_heading)  # 単体では見出し扱いされる字面
        chunks = chunk_structure(
            _entries(("heading", "Real Heading"), ("text", LONG),
                     ("text", looks_like_heading), ("text", LONG)),
            "src",
        )
        assert all(c["heading"] == "Real Heading" for c in chunks)
        assert any(looks_like_heading in c["text"] for c in chunks)  # 本文側に残る


class TestHeuristicPath:
    """SoundQuest経路: heading型が無ければ従来のheuristicで境界を推定する。"""

    def test_short_line_without_kuten_is_heading(self):
        chunks = chunk_structure(
            _entries(("text", "コードの機能"), ("text", LONG)), "src"
        )
        assert chunks[0]["heading"] == "コードの機能"

    def test_audio_and_image_entries_never_headings(self):
        assert not _looks_like_heading("短いテキスト", entry_type="audio")
        assert not _looks_like_heading("短いテキスト", entry_type="image")


class TestChunkContract:
    """両経路共通の出力契約。"""

    def test_chunk_index_contiguous_from_zero(self):
        chunks = chunk_structure(
            _entries(("heading", "A"), ("text", LONG * 6),   # 960字 > CHUNK_CHARS → 分割
                     ("heading", "B"), ("text", LONG)),
            "src",
        )
        assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
        assert all(c["source"] == "src" for c in chunks)

    def test_long_section_split_keeps_breadcrumb_on_every_piece(self):
        long_text = LONG * 8  # 1280字 > CHUNK_CHARS(800)
        chunks = chunk_structure(
            _entries(("heading", "Long"), ("text", long_text)), "src"
        )
        assert len(chunks) >= 2
        assert all(c["text"].startswith("[Long]\n") for c in chunks)

    def test_short_section_merges_forward(self):
        """MIN_SECTION_CHARS未満のセクションは次のセクションへ吸収され、
        breadcrumbは ' > ' で連結される。"""
        short = "短い導入。"  # < 100字
        chunks = chunk_structure(
            _entries(("heading", "Tiny"), ("text", short),
                     ("heading", "Main"), ("text", LONG)),
            "src",
        )
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Tiny > Main"
        assert short in chunks[0]["text"]

    def test_config_assumptions(self):
        """テストが前提にする設定値が変わったら気づけるように固定する。"""
        assert config.CHUNK_CHARS == 800
        assert config.MIN_SECTION_CHARS == 100
