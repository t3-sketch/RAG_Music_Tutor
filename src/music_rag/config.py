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

# ハイブリッド検索用 collection（named vectors: dense + sparse）。dense-only の
# COLLECTIONS とは別物で、ingest 側で sparse を持たせた再 upsert が前提。
HYBRID_COLLECTION = os.getenv("HYBRID_COLLECTION", "music_theory_hybrid")

# 検索層は条件C（dense+sparse ハイブリッド × query expansion）を採用（2026-07-21 実験）。
# 66問 recall@5: Base 0.599 → C 0.674、single 20問は 1.000、AND 27問 0.352→0.426。
# 有意ではない（p=0.099）が全指標で一貫して正なので採用。切り戻しは env で個別に。
ENABLE_HYBRID = os.getenv("ENABLE_HYBRID", "true").lower() in ("true", "1", "yes")
ENABLE_QE = os.getenv("ENABLE_QE", "true").lower() in ("true", "1", "yes")
# 各 Prefetch ブランチの取得数。RRF で融合してから top_k に切る。
HYBRID_PREFETCH_LIMIT = int(os.getenv("HYBRID_PREFETCH_LIMIT", "50"))

# --- Qdrant（Docker, http://localhost:6333）---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", COLLECTION_NAME)

# Qdrant Cloud（マネージド）へのコピー先。corpus転送スクリプト（非公開）が参照する。
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

# --- OpenRouter（OpenAI互換。1キーで Kimi / Qwen / MiniMax / DeepSeek をモデル名だけで切替）---
# judge を Gemini 以外のプロバイダに逃がす用途がメイン（生成=Gemini との self-preference bias 回避）。
# 実測(2026-07-16)の judge 実額/60問: minimax/minimax-m2.5 $0.23 / deepseek-v4-flash $0.11 /
# kimi-k2.5 $0.82 / qwen3-235b-a22b-2507 $0.14。どれも誤差なので値段で選ぶ意味はない。
# .env 側の表記ゆれ（OPEN_ROUTER_API_KEY）も拾う。
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY") or ""
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.5")

# --- 生成層プロバイダ切替（backlog「生成LLMを選べる」の最小版。env LLM_PROVIDER で切替）---
# gemini / nvidia / openrouter。既定は gemini。
# NVIDIA無料枠は実測で使用不能（2026-07-16: 112tok応答に223秒、3回中2回が120秒でタイムアウト）。
# 注意: gemini 無料枠は RPD 上限が厳しい（失敗リクエストもRPDを消費する）。連続評価では枯渇注意。
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEN_GEMINI_MODEL = os.getenv("GEN_GEMINI_MODEL", "gemini-3.5-flash-lite")

# query expansion 専用モデル。GEMINI_MODEL（RAGAS judge 兼用）とは意図的に分離する
# ── judge を差し替えたときに本番の検索が黙って変わるのを防ぐため。
# 3.5-flash-lite への差し替えは実験で不採用（全指標で 3.1 に劣後。docs/retrieval-experiment-results-qe35.md）。
QE_GEMINI_MODEL = os.getenv("QE_GEMINI_MODEL", "gemini-3.1-flash-lite")
# QE は検索前の1往復。ユーザーを待たせる区間なので生成層(90s)より短く切って諦める。
QE_TIMEOUT_SEC = float(os.getenv("QE_TIMEOUT_SEC", "10"))
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

# 取得チャンクの原文表示（デバッグ用）。公開デモでは原文を露出させないため既定 false。
# 出典は記事タイトル＋リンクのみで示し、本文は元サイトへ送客する方針（CLAUDE.md §8）。
SHOW_DEBUG_CHUNKS = os.getenv("SHOW_DEBUG_CHUNKS", "false").lower() in ("true", "1", "yes")

# --- 音声URL入力（YouTube / ニコニコ動画）---
# yt-dlpによるダウンロードは各サービスの利用規約に抵触しうるため、
# ローカル個人利用・研究目的限定の機能。公開デプロイ時は false にして無効化する。
ENABLE_URL_INPUT = os.getenv("ENABLE_URL_INPUT", "true").lower() in ("true", "1", "yes")
# ダウンロード前にメタデータで弾く動画長の上限（秒）。BTC推論時間を実用範囲に抑える。
MAX_AUDIO_DURATION_SEC = int(os.getenv("MAX_AUDIO_DURATION_SEC", "600"))
