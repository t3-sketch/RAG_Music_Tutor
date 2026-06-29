"""音響解析モジュール。

2系統の入力をサポートする:
  1. ローカル音声ファイル（mp3/wav/flac）  -> Librosa で特徴抽出
  2. web上の楽曲URL（YouTube/ニコニコ等）  -> Songle API で解析結果を取得

どちらも最終的に「Claude に渡す日本語テキスト」へ変換する。
"""
from __future__ import annotations

import numpy as np
import requests

import config

# --- 音名・調推定用プロファイル（Krumhansl-Schmuckler）---
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


# =========================================================
#  1. ローカルファイル: Librosa
# =========================================================
def extract_local(path: str) -> dict:
    """ローカル音声ファイルから音響特徴を抽出する。"""
    import librosa  # 重いので関数内 import

    y, sr = librosa.load(path, sr=22050, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    chroma = librosa.feature.chroma_cens(y=y, sr=sr).mean(axis=1)
    key = _estimate_key(chroma)

    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())

    return {
        "bpm": round(tempo, 1),
        "key": key,
        "brightness_hz": round(centroid, 0),
        "energy_rms": round(rms, 4),
        "zcr": round(zcr, 4),
        "duration_sec": round(len(y) / sr, 1),
    }


def _estimate_key(chroma_mean: np.ndarray) -> str:
    """12次元クロマから最も相関の高い調（major/minor）を返す。"""
    best_name, best_score = "不明", -2.0
    for i in range(12):
        for profile, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            rolled = np.roll(profile, i)
            score = float(np.corrcoef(rolled, chroma_mean)[0, 1])
            if score > best_score:
                best_score = score
                best_name = f"{PITCHES[i]} {mode}"
    return best_name


def describe_local(f: dict) -> str:
    """Librosa の特徴を日本語の説明テキストにする。"""
    brightness = "明るい" if f["brightness_hz"] > 2500 else "落ち着いた"
    energy = "強い" if f["energy_rms"] > 0.05 else "穏やか"
    return (
        f"- 解析方法: Librosa（ローカル音源の信号解析）\n"
        f"- 推定BPM: {f['bpm']}\n"
        f"- 推定キー: {f['key']}\n"
        f"- 長さ: {f['duration_sec']} 秒\n"
        f"- 音色の明るさ: {brightness}（スペクトル重心 {f['brightness_hz']:.0f} Hz）\n"
        f"- エネルギー感: {energy}（RMS {f['energy_rms']}）\n"
        f"- ノイズ/打楽器成分の目安: ZCR {f['zcr']}\n"
        f"※ キーとBPMは信号からの統計的推定であり、誤差を含む。"
    )


# =========================================================
#  2. web上の楽曲URL: Songle API（産総研）
# =========================================================
def fetch_songle(url: str) -> dict:
    """
    Songle から楽曲解析結果（基本情報・コード・ビート・サビ構造）を取得する。
    対象URLが Songle 側で事前解析済みである必要がある。
    未解析の場合は songle.jp 上で解析申請しておくこと。
    """
    params = {"url": url}
    headers = {}
    if config.SONGLE_API_TOKEN:
        # 認証方式はダッシュボードで要確認（ヘッダ or クエリ）。
        headers["X-Songle-Api-Token"] = config.SONGLE_API_TOKEN

    def _get(endpoint: str):
        r = requests.get(
            f"{config.SONGLE_API_BASE}/{endpoint}",
            params=params,
            headers=headers,
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    song = _get("song")
    chords = _list(_get("song/chord"), "chords")
    beats = _list(_get("song/beat"), "beats")
    chorus = _list(_get("song/chorus"), "chorusSegments")

    return {
        "title": _dig(song, "title") or "（不明）",
        "artist": _dig(song, "artist", "name") or "（不明）",
        "duration_sec": round((_dig(song, "duration") or 0) / 1000, 1),
        "bpm": _bpm_from_beats(beats),
        "chord_progression": _chord_progression(chords),
        "unique_chords": _unique_chords(chords),
        "n_chorus": len(chorus),
    }


def _list(data, key: str) -> list:
    """レスポンスの形ゆれを吸収して目的のリストを取り出す。"""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if key in data and isinstance(data[key], list):
            return data[key]
        for v in data.values():
            if isinstance(v, dict) and key in v and isinstance(v[key], list):
                return v[key]
            if isinstance(v, list):
                return v
    return []


def _dig(data, *keys):
    cur = data
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _bpm_from_beats(beats: list) -> float | None:
    """連続する拍の間隔の中央値から BPM を推定する。"""
    starts = [b.get("start") for b in beats if isinstance(b, dict) and "start" in b]
    starts = sorted(s for s in starts if s is not None)
    if len(starts) < 2:
        return None
    diffs = np.diff(starts)
    median = float(np.median(diffs))
    if median <= 0:
        return None
    # start がミリ秒で来る場合（間隔が大きい）は秒に換算
    interval_sec = median / 1000 if median > 10 else median
    return round(60.0 / interval_sec, 1)


def _chord_progression(chords: list, limit: int = 24) -> list[str]:
    """連続する同名コードをまとめてコード進行のシーケンスにする。"""
    names = [c.get("name") for c in chords if isinstance(c, dict) and c.get("name")]
    seq, prev = [], None
    for name in names:
        if name != prev:
            seq.append(name)
            prev = name
    return seq[:limit]


def _unique_chords(chords: list) -> list[str]:
    seen = []
    for c in chords:
        name = c.get("name") if isinstance(c, dict) else None
        if name and name not in seen:
            seen.append(name)
    return seen


def describe_songle(d: dict) -> str:
    """Songle の解析結果を日本語の説明テキストにする。"""
    prog = " → ".join(d["chord_progression"]) if d["chord_progression"] else "（取得なし）"
    uniq = ", ".join(d["unique_chords"]) if d["unique_chords"] else "（取得なし）"
    bpm = d["bpm"] if d["bpm"] is not None else "（取得なし）"
    return (
        f"- 解析方法: Songle API（産総研の音楽理解技術による解析結果）\n"
        f"- 曲名 / アーティスト: {d['title']} / {d['artist']}\n"
        f"- 長さ: {d['duration_sec']} 秒\n"
        f"- 推定BPM: {bpm}\n"
        f"- コード進行（冒頭）: {prog}\n"
        f"- 使用コード一覧: {uniq}\n"
        f"- サビ（繰り返し区間）の数: {d['n_chorus']}"
    )
