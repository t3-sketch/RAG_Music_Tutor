"""プロジェクト全体の設定を一元管理するモジュール。"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- データパス ---
# 既定は cwd 基準の ./data（コマンドは repo ルートから実行する規約）。
# 別の場所から実行する場合は MUSIC_RAG_DATA_DIR で上書きする。
DATA_DIR = Path(os.getenv("MUSIC_RAG_DATA_DIR", "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
# オープンライセンスcorpus（OMT等）。取得スクリプト群はgitignoreだが、
# パス・collection名自体は中立な設定値なのでここで一元管理する。
RAW_OMT_DIR = DATA_DIR / "raw_omt"
RAW_OPEN_DIR = DATA_DIR / "raw_open"
EVAL_DIR = DATA_DIR / "eval"
GATED_REPORT = DATA_DIR / "reports" / "gated_report.txt"

# --- ベクトルDB ---
# chunking戦略ごとの collection。evaluation.py が戦略比較にも使う。
COLLECTIONS = {
    "fixed": "music_theory",
    "structure": "music_theory_structure",
}

# 本番のchunking戦略。RAGAS評価（2026-07-02、context_precision +0.134）を受けて
# structure をデフォルト採用。A/B比較時は環境変数 CHUNK_STRATEGY=fixed で切替。
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "structure")
COLLECTION_NAME = COLLECTIONS[CHUNK_STRATEGY]

# --- Qdrant（Docker, http://localhost:6333）---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", COLLECTION_NAME)

# Qdrant Cloud（マネージド）へのコピー先。scripts/copy_collection.py が参照する。
# ローカルQdrant → Cloud への公開用collection転送でのみ使い、本番の検索経路
# （QDRANT_URL）とは分離する。URL は :6333 を付けない（Cloud は 443/https 受け）。
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL") or None
QDRANT_CLOUD_API_KEY = os.getenv("QDRANT_CLOUD_API_KEY") or None
# オープンcorpus用collection（SoundQuest系collectionとは分離してA/B比較可能に保つ）
OPEN_COLLECTION = os.getenv("OPEN_COLLECTION", "music_theory_open")
# BGE-M3 の dense ベクトル次元。collection 作成時の次元と必ず一致させる。
EMBED_DIM = 1024
# 距離関数（cosine 固定）
DISTANCE = "cosine"

# --- モデル ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
# RAGAS judge（評価層）専用。生成層とは別モデルに分離（self-preference bias回避）。
# evaluation.py からのみ参照される（experiments/evaluation.py 参照）。
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Gemini の OpenAI互換エンドポイント（生成層で gemini を使う場合に共有）。
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# --- NVIDIA Build（OpenAI互換API）---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.3-70b-instruct")

# --- 生成層プロバイダ切替（backlog「生成LLMを選べる」の最小版。env LLM_PROVIDER で切替）---
# gemini / nvidia。既定は gemini（NVIDIA endpoint が応答遅延・ハングするため切替、2026-07-13）。
# 注意: gemini 無料枠は RPD 上限が厳しい（失敗リクエストもRPDを消費する）。連続評価では枯渇注意。
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEN_GEMINI_MODEL = os.getenv("GEN_GEMINI_MODEL", "gemini-3.5-flash")
# 生成呼び出しのタイムアウト（秒）。未指定だとSDK既定=600秒でハングし得るため明示。
LLM_TIMEOUT_SEC = int(os.getenv("LLM_TIMEOUT_SEC", "90"))

# --- チャンク分割（文字数ベース。日本語教材を想定）---
CHUNK_CHARS = 800
CHUNK_OVERLAP = 120
MIN_SECTION_CHARS = 100

# --- 検索・生成 ---
TOP_K = 5
MAX_TOKENS = 1500

# 音声入力UI全体のゲート（MVP限定公開ではテキストQAに絞るため false にする）
ENABLE_AUDIO_INPUT = os.getenv("ENABLE_AUDIO_INPUT", "true").lower() in ("true", "1", "yes")

# --- 音声URL入力（YouTube / ニコニコ動画）---
# yt-dlpによるダウンロードは各サービスの利用規約に抵触しうるため、
# ローカル個人利用・研究目的限定の機能。公開デプロイ時は false にして無効化する。
ENABLE_URL_INPUT = os.getenv("ENABLE_URL_INPUT", "true").lower() in ("true", "1", "yes")
# ダウンロード前にメタデータで弾く動画長の上限（秒）。BTC推論時間を実用範囲に抑える。
MAX_AUDIO_DURATION_SEC = int(os.getenv("MAX_AUDIO_DURATION_SEC", "600"))
