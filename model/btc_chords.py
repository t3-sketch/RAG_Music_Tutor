"""BTC-ISMIR19のコードラベル定義（large_voca = 170クラス）。

Vendored from jayg996/BTC-ISMIR19 (MIT License, Copyright (c) 2019 Jonggwon Park)
元リポジトリ: https://github.com/jayg996/BTC-ISMIR19
元論文: "A Bi-Directional Transformer for Musical Chord Recognition" (ISMIR 2019)
ライセンス全文: model/LICENSE_BTC-ISMIR19

出典: https://github.com/jayg996/BTC-ISMIR19 utils/mir_eval_modules.py の idx2voca_chord()
改変内容: mir_evalへの依存を避けるため、ラベル生成に必要な idx2voca_chord() 相当の
処理のみを抜粋・移植し、型ヒントを付与した。同ファイル内の他の関数（特徴量抽出等）は含まない。
"""

_ROOT_LIST = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_QUALITY_LIST = [
    "min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7",
    "maj7", "7", "dim7", "hdim7", "sus2", "sus4",
]


def idx2voca_chord() -> dict[int, str]:
    """モデル出力インデックス(0-169) → BTC生ラベル文字列（例: "C", "C#:min", "N", "X"）。"""
    mapping: dict[int, str] = {169: "N", 168: "X"}
    for i in range(168):
        root = _ROOT_LIST[i // 14]
        quality = _QUALITY_LIST[i % 14]
        mapping[i] = root if quality == "maj" else f"{root}:{quality}"
    return mapping
