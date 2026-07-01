"""プロジェクト全体の設定を一元管理するモジュール。"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- ベクトルDB ---
COLLECTION_NAME = "music_theory"

# 評価実験用の collection（chunking戦略ごと）。
# 本番の QDRANT_COLLECTION とは別物。evaluation.py が戦略比較に使う。
COLLECTIONS = {
    "fixed": "music_theory",
    "structure": "music_theory_structure",
}

# --- Qdrant（Docker, http://localhost:6333）---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", COLLECTION_NAME)
# BGE-M3 の dense ベクトル次元。collection 作成時の次元と必ず一致させる。
EMBED_DIM = 1024
# 距離関数（cosine 固定）
DISTANCE = "cosine"

# --- モデル ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Songle API ---
SONGLE_API_TOKEN = os.getenv("SONGLE_API_TOKEN", "")
# 注意: ベースURLと認証方式は取得済みトークンのダッシュボードで要確認。
# 必要に応じてここを書き換えてください。
SONGLE_API_BASE = "https://widget.songle.jp/api/v1"

# --- チャンク分割（文字数ベース。日本語教材を想定）---
CHUNK_CHARS = 800
CHUNK_OVERLAP = 120
MIN_SECTION_CHARS = 100

# --- 検索・生成 ---
TOP_K = 5
MAX_TOKENS = 1500
