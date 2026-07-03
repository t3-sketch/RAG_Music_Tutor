# music-rag Streamlit app（公開デプロイ準備。デプロイ自体は権利者許諾後）
#
# 注意: torch + FlagEmbedding を含むためイメージは数GB級になる。
# BGE-M3 の重み（約2GB）はイメージに含めず、初回起動時に HF Hub から
# ダウンロードされる（HF_HOME を volume にすると再取得を避けられる）。
FROM python:3.11-slim

# librosa / audioread が音声デコードに使うシステムライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 依存レイヤーをソースと分けてキャッシュを効かせる
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# ソース一式（music_rag パッケージ + UI）
COPY src/ src/
COPY apps/ apps/
RUN uv sync --frozen --no-dev

# データパス（corpus はイメージに含めない。volume でマウントする）
ENV MUSIC_RAG_DATA_DIR=/app/data

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "apps/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
