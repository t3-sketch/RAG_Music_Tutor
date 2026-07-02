"""audio.py

Phase 2 MVP: 音声ファイル単体の解析関数群。
chroma特徴量をdetect_key/detect_chordsで共有する処理軸分割（案B, 2026-07-01決定）。

スコープ外（将来課題）:
  - メロディ/F0解析
  - segment分割（Aメロ/Bメロ/サビ）
  - テンションコード（9th/11th/13th）
  - 音源分離
"""

from __future__ import annotations

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# コードテンプレート（MajMin + 7th語彙）
# ---------------------------------------------------------------------------

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_CHORD_INTERVALS = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "7": [0, 4, 7, 10],      # dominant 7th
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
}


def _build_chord_templates() -> dict[str, np.ndarray]:
    """root x quality の全組み合わせで単位ベクトル化されたバイナリchromaテンプレートを作る。"""
    templates: dict[str, np.ndarray] = {}
    for quality, intervals in _CHORD_INTERVALS.items():
        for root_idx, root_name in enumerate(_PITCH_CLASSES):
            vec = np.zeros(12)
            for interval in intervals:
                vec[(root_idx + interval) % 12] = 1.0
            vec = vec / np.linalg.norm(vec)  # コード種によってノート数が違うため正規化
            templates[f"{root_name} {quality}"] = vec
    return templates


_CHORD_TEMPLATES = _build_chord_templates()

# Krumhansl-Schmuckler key profile
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _load_and_chroma(path: str, hop_length: int = 2048) -> tuple[np.ndarray, int]:
    """音声を読み込み、chroma特徴量を計算する。

    Returns:
        chroma: shape (12, n_frames)
        sr: サンプルレート（時刻変換に使うため呼び出し側に返す）
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    return chroma, sr


def detect_tempo_beats(path: str) -> dict:
    """テンポ（BPM）とビート位置（秒）を検出する。

    Returns:
        {"tempo": float, "beats": [float, ...]}
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])  # librosa 0.10+ は配列を返すため先頭要素を取る
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    return {"tempo": float(tempo), "beats": beat_times}


def detect_key(chroma: np.ndarray) -> str:
    """Krumhansl-Schmuckler相関でグローバルキーを推定する。

    Args:
        chroma: shape (12, n_frames), _load_and_chroma()の出力。

    Returns:
        例: "C maj", "A min"
    """
    mean_chroma = chroma.mean(axis=1)

    best_score = -np.inf
    best_key = "C maj"
    for root_idx, root_name in enumerate(_PITCH_CLASSES):
        for profile, quality in [(_MAJOR_PROFILE, "maj"), (_MINOR_PROFILE, "min")]:
            rotated = np.roll(profile, root_idx)
            score = np.corrcoef(mean_chroma, rotated)[0, 1]
            if score > best_score:
                best_score = score
                best_key = f"{root_name} {quality}"
    return best_key


def _btc_label_to_common(label: str) -> str:
    """BTCの生ラベル（例: "C", "C#:min", "C:7", "N", "X"）を既存の"root quality"形式に変換する。"""
    if label in ("N", "X"):
        return "N"
    if ":" in label:
        root, quality = label.split(":", 1)
        return f"{root} {quality}"
    return f"{label} maj"


def detect_chords(
    path: str,
    chroma: np.ndarray | None = None,
    sr: int | None = None,
    hop_length: int = 2048,
    frame_duration: float = 1.0,
    device: str = "cpu",
) -> list[dict]:
    """BTC-ISMIR19モデル（Bi-directional Transformer for Chord recognition）でコード進行を推定する。

    モデルのロード・推論に失敗した場合（torch未インストール、重みファイル欠損等）は
    _detect_chords_template（テンプレートマッチング）にフォールバックする。

    Args:
        path: 音声ファイルパス。
        chroma, sr: フォールバック時に使うchroma特徴量。省略時はpathから再計算する。
        hop_length, frame_duration: フォールバック（テンプレートマッチング）用パラメータ。
        device: BTCモデルの実行デバイス。Apple Siliconでは推論時間の大半がCQT特徴量抽出
            （librosa, CPU処理）に占められモデル計算自体は高速なため、"cpu"固定で問題ない
            （2026-07-01 実機検証）。

    Returns:
        [{"start": float, "end": float, "chord": str}, ...]
        同一コードが連続する区間はマージ済み。
    """
    try:
        from model.btc_infer import recognize_chords_btc

        raw_chords = recognize_chords_btc(path, device=device)
        converted = [
            (c["start"], c["end"], _btc_label_to_common(c["chord"]))
            for c in raw_chords
        ]
        return _merge_adjacent_chords(converted)
    except (ImportError, OSError, RuntimeError):
        if chroma is None or sr is None:
            chroma, sr = _load_and_chroma(path, hop_length=hop_length)
        return _detect_chords_template(chroma, sr, hop_length=hop_length, frame_duration=frame_duration)


def _detect_chords_template(
    chroma: np.ndarray,
    sr: int,
    hop_length: int = 2048,
    frame_duration: float = 1.0,
) -> list[dict]:
    """固定長ウィンドウのテンプレートマッチングでコード進行を推定する（detect_chordsのフォールバック実装）。

    Args:
        chroma: shape (12, n_frames), _load_and_chroma()の出力。
        sr: サンプルレート。
        hop_length: chroma計算時のhop_length（時刻変換に必要）。
        frame_duration: コード推定の窓幅（秒）。窓内のchromaフレームを平均してから判定。

    Returns:
        [{"start": float, "end": float, "chord": str}, ...]
        同一コードが連続する区間はマージ済み。
    """
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    total_duration = frame_times[-1] if len(frame_times) else 0.0

    raw_chords: list[tuple[float, float, str]] = []
    t = 0.0
    while t < total_duration:
        t_end = min(t + frame_duration, total_duration)
        mask = (frame_times >= t) & (frame_times < t_end)
        if mask.any():
            window_chroma = chroma[:, mask].mean(axis=1)
            chord = _match_chord_template(window_chroma)
            raw_chords.append((t, t_end, chord))
        t = t_end

    return _merge_adjacent_chords(raw_chords)


def _match_chord_template(chroma_vec: np.ndarray) -> str:
    norm = np.linalg.norm(chroma_vec)
    if norm == 0:
        return "N"  # 無音・エネルギーなし
    normalized = chroma_vec / norm

    best_score = -np.inf
    best_chord = "N"
    for name, template in _CHORD_TEMPLATES.items():
        score = np.dot(normalized, template)  # cosine similarity
        if score > best_score:
            best_score = score
            best_chord = name
    return best_chord


def _merge_adjacent_chords(raw_chords: list[tuple[float, float, str]]) -> list[dict]:
    if not raw_chords:
        return []
    merged = [{"start": raw_chords[0][0], "end": raw_chords[0][1], "chord": raw_chords[0][2]}]
    for start, end, chord in raw_chords[1:]:
        if chord == merged[-1]["chord"]:
            merged[-1]["end"] = end
        else:
            merged.append({"start": start, "end": end, "chord": chord})
    return merged


def analyze(path: str) -> dict:
    """Phase 2 MVPの全解析をまとめて実行する接着関数。

    Returns:
        {
            "tempo": float,
            "beats": [float, ...],
            "key": str,              # 例: "C maj"
            "chords": [{"start": float, "end": float, "chord": str}, ...],
        }
    """
    chroma, sr = _load_and_chroma(path)
    tempo_beats = detect_tempo_beats(path)  # 下の注記参照: 音声を2回ロードしている
    key = detect_key(chroma)
    chords = detect_chords(path, chroma=chroma, sr=sr)

    return {
        "tempo": tempo_beats["tempo"],
        "beats": tempo_beats["beats"],
        "key": key,
        "chords": chords,
    }
