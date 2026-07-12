"""audio.describe の characterization テスト（純粋関数、音声ファイル不要）。

describe は analyze() の結果を生成層プロンプト用の日本語に変換する。
「LLMに何を見せるか」の契約なので、フォーマット変更は生成品質に直結する。
"""
from music_rag.audio import describe


def _chord(name: str, start: float, end: float) -> dict:
    return {"chord": name, "start": start, "end": end}


class TestDescribe:
    def test_tempo_rounded_and_key_shown(self):
        out = describe({"tempo": 128.4, "key": "D# major", "chords": []})
        assert "- テンポ: 約 128 BPM" in out
        assert "- 推定キー: D# major" in out

    def test_no_chords_gives_only_two_lines(self):
        out = describe({"tempo": 120.0, "key": "C major", "chords": []})
        assert out.count("\n") == 1  # テンポ・キーの2行のみ

    def test_main_chords_sorted_by_total_duration(self):
        """主要コードは出現時間の合計の降順（登場順ではない）。"""
        out = describe({
            "tempo": 120.0, "key": "C major",
            "chords": [
                _chord("C", 0, 1), _chord("G", 1, 5), _chord("C", 5, 6),
            ],
        })
        # G=4.0秒 > C=2.0秒
        line = next(l for l in out.split("\n") if l.startswith("- 主要コード"))
        assert line.index("G（4.0秒）") < line.index("C（2.0秒）")

    def test_no_chord_segments_excluded(self):
        """"N"(no chord)区間は主要コードにも進行にも出さない。"""
        out = describe({
            "tempo": 120.0, "key": "C major",
            "chords": [_chord("N", 0, 10), _chord("Am", 10, 11)],
        })
        assert "N（" not in out
        assert "N →" not in out and "→ N" not in out
        assert "Am" in out

    def test_progression_truncated_at_48_segments(self):
        chords = [_chord(f"C", i, i + 1) for i in range(50)]
        out = describe({"tempo": 120.0, "key": "C major", "chords": chords})
        prog_line = next(l for l in out.split("\n") if l.startswith("- コード進行"))
        assert prog_line.endswith("（以降省略）")
        assert prog_line.count("→") == 47  # 48要素 = 矢印47本

    def test_progression_not_truncated_when_short(self):
        chords = [_chord("C", 0, 1), _chord("G", 1, 2)]
        out = describe({"tempo": 120.0, "key": "C major", "chords": chords})
        assert "（以降省略）" not in out
        assert "C → G" in out
