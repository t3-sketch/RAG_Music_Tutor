"""BTC-ISMIR19 (large_voca) モデルによるコード認識の推論ラッパー。

出典: https://github.com/jayg996/BTC-ISMIR19 (MIT License, test.py相当の処理を関数化)
学習用コード(train.py等)は含まない。そのため mir_eval / pretty_midi / pyrubberband / pyyaml
は依存に加えていない（推論のみなら不要と実機検証済み、2026-07-01）。

BTC-ISMIR19オリジナルコードとの差分（2019年当時のAPIとの非互換の修正）:
  - utils/hparams.py 経由のYAML読み込みをやめ、run_config.yamlの内容をここに直接定数化
    (PyYAML 6.xで yaml.load() の Loader引数必須化に対応する必要がなくなる)
  - transformer_modules.py: np.float -> float (numpy 2.x で削除されたエイリアス)
  - torch.load(..., map_location=device): 元コードはCUDA前提でCPU環境だと例外になる
  - torch.load(..., weights_only=False): PyTorch 2.6+のデフォルト変更でnumpyスカラーを
    含む旧チェックポイントの読み込みに失敗するため明示指定（配布元は公式リポジトリ同梱の
    信頼できるファイルのため許容）
"""

from __future__ import annotations

import os

import librosa
import numpy as np
import torch

from model.btc_chords import idx2voca_chord
from model.btc_model import BTC_model

_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "btc_large_voca.pt")

_CONFIG = {
    "mp3": {"song_hz": 22050, "inst_len": 10.0},
    "feature": {"n_bins": 144, "bins_per_octave": 24, "hop_length": 2048},
    "model": {
        "feature_size": 144,
        "timestep": 108,
        "num_chords": 170,
        "input_dropout": 0.2,
        "layer_dropout": 0.2,
        "attention_dropout": 0.2,
        "relu_dropout": 0.2,
        "num_layers": 8,
        "num_heads": 4,
        "hidden_size": 128,
        "total_key_depth": 128,
        "total_value_depth": 128,
        "filter_size": 128,
        "loss": "ce",
        "probs_out": False,
    },
}

_IDX_TO_CHORD = idx2voca_chord()

_model_cache: dict[str, tuple] = {}


def _load_model(device: str) -> tuple:
    if device in _model_cache:
        return _model_cache[device]

    torch_device = torch.device(device)
    model = BTC_model(config=_CONFIG["model"]).to(torch_device)
    checkpoint = torch.load(_WEIGHTS_PATH, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    cached = (model, checkpoint["mean"], checkpoint["std"], torch_device)
    _model_cache[device] = cached
    return cached


def _audio_to_cqt_feature(path: str) -> tuple[np.ndarray, float]:
    """音声をBTC用のlog-CQT特徴量に変換する。

    Returns:
        feature: shape (n_frames, n_bins)
        feature_per_second: 1フレームあたりの秒数
    """
    song_hz = _CONFIG["mp3"]["song_hz"]
    inst_len_samples = int(_CONFIG["mp3"]["inst_len"] * song_hz)
    n_bins = _CONFIG["feature"]["n_bins"]
    bins_per_octave = _CONFIG["feature"]["bins_per_octave"]
    hop_length = _CONFIG["feature"]["hop_length"]

    wav, sr = librosa.load(path, sr=song_hz, mono=True)

    chunks = []
    cursor = 0
    while cursor + inst_len_samples < len(wav):
        chunk = librosa.cqt(
            wav[cursor:cursor + inst_len_samples], sr=sr,
            n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length,
        )
        chunks.append(chunk)
        cursor += inst_len_samples
    chunks.append(librosa.cqt(
        wav[cursor:], sr=sr,
        n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length,
    ))

    feature = np.concatenate(chunks, axis=1)
    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = _CONFIG["mp3"]["inst_len"] / _CONFIG["model"]["timestep"]
    return feature.T, feature_per_second


def recognize_chords_btc(path: str, device: str = "cpu") -> list[dict]:
    """BTC-ISMIR19 (large_voca)でコード進行を推定する。

    Returns:
        [{"start": float, "end": float, "chord": str}, ...]
        chordはBTC生ラベル（例: "C", "C#:min", "C:7", "N", "X"）のまま返す。
        呼び出し側で既存フォーマットへの変換を行うこと。
    """
    model, mean, std, torch_device = _load_model(device)

    feature, time_unit = _audio_to_cqt_feature(path)
    feature = (feature - mean) / std

    n_timestep = _CONFIG["model"]["timestep"]
    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    intervals: list[dict] = []
    start_time = 0.0
    with torch.no_grad():
        feature_t = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(torch_device)
        prev_chord = None
        for t in range(num_instance):
            self_attn_output, _ = model.self_attn_layers(
                feature_t[:, n_timestep * t:n_timestep * (t + 1), :]
            )
            prediction, _ = model.output_layer(self_attn_output)
            prediction = prediction.squeeze()
            for i in range(n_timestep):
                global_i = n_timestep * t + i
                if prev_chord is None:
                    prev_chord = prediction[i].item()
                    continue
                if prediction[i].item() != prev_chord:
                    end_time = time_unit * global_i
                    intervals.append({
                        "start": start_time, "end": end_time,
                        "chord": _IDX_TO_CHORD[prev_chord],
                    })
                    start_time = end_time
                    prev_chord = prediction[i].item()
                if t == num_instance - 1 and i + num_pad == n_timestep:
                    if start_time != time_unit * global_i:
                        intervals.append({
                            "start": start_time, "end": time_unit * global_i,
                            "chord": _IDX_TO_CHORD[prev_chord],
                        })
                    break

    return intervals
