---
title: Music RAG
emoji: 🎵
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.58.0
app_file: apps/streamlit_app.py
python_version: "3.11"
pinned: false
short_description: 日本語の音楽理論教材を根拠に出典つきで答えるRAG
---

# music-rag

日本語の音楽理論教材コーパスを根拠に、コード進行・メロディ・リズムに関する質問へ日本語で解説する RAG システムです。
ユーザーの質問（＋任意で楽曲の音響特徴）に対し、教材から関連箇所を検索し、それを根拠に LLM が解説を生成します。

> **コーパスの2系統**:
> - **ローカル/フル版**: SoundQuest（soundquest.jp）の記事コーパス。著作権は原著者に帰属し、利用許諾を打診中のため**公開デプロイには使いません**（コーパス本体 `data/` もリポジトリ非同梱）。
> - **公開デモ版**: オープンライセンス教材（Open Music Theory 等）のコーパス（Qdrant `music_theory_open`）。権利上問題なく共有できるため、**公開デモ（Hugging Face Spaces）はこちらを検索対象**にしています。
>
> コードとアーキテクチャは両系統で共通で、`QDRANT_COLLECTION` で検索先を切り替えるだけです。

---

## Demo

> 質問を入力すると、教材から関連箇所を検索し、それを根拠に日本語の解説を生成します。
> 解説の出典となった教材を併記します。

![demo](docs/demo.png)

---

## 公開デモ（デプロイ構成）

公開デモは **Hugging Face Spaces（Streamlit SDK, 無料CPU）** にデプロイしています。
無料CPU枠でも約16GB RAM あり、埋め込み（BGE-M3）・音響解析（librosa + BTC）を含めた
全処理を Space 内で実行できます。GitHub の `main` への push で
[GitHub Actions](.github/workflows/hf-sync.yml) が自動で Space へ反映します。

| 層 | 公開デモでの構成 |
| --- | --- |
| 検索先 | Qdrant Cloud（`music_theory_open` = オープンライセンス教材） |
| 埋め込み | ローカル BGE-M3（FlagEmbedding。Space 内で実行） |
| 生成 | NVIDIA Build（`meta/llama-3.3-70b-instruct`、OpenAI互換API） |
| 音声入力 | ファイルアップロードは有効。URL入力（yt-dlp）は利用規約配慮で無効（`ENABLE_URL_INPUT=false`） |
- **生成と評価 judge はプロバイダごと分離**: 生成は NVIDIA、RAGAS の judge は Gemini
  （`gemini-3.1-flash-lite`）。self-preference bias を避けつつ、無料枠の
  レート制限（Gemini の RPD 枯渇）を評価と本番で独立させています。

---

## 開発背景

「音楽理論は作曲に必要ない」と有名インフルエンサーが主張する投稿を見かけるたびに、その根拠の薄さに疑問を感じていました。実際には、著名なプロの作曲家はほぼ例外なく音楽理論を学んでいます。学んでいない人もいますが、それは長年積み重ねた音楽経験があるからこそであり、その経験則も突き詰めれば結局は「音楽理論的なもの」へと収斂していきます。

そして実際に学んでみると、今まで何気なく聴いていた楽曲の聴こえ方が変わり、コード進行の仕掛けに気づいて何倍も面白く感じられ、新たな発見が連続する世界が広がります。一方で私自身も「どこから学べばいいかわからない」という壁に直面しました。学習で最も頼りにしたのが SoundQuest という日本語の音楽理論サイトです。体系的で質が高い反面、記事数が膨大で、「今知りたいこの概念」にすぐ辿り着くのが難しい状況でした。

認知科学的にも、自ら問いを立てて答えを生成しようとするときの負荷（生成効果）こそが、効率の良い学びにつながるとされています。質問を投げれば該当箇所を根拠付きで返してくれる形にすれば、音楽理論はもっと面白く、もっと身近になる。そう考えてこのプロジェクトを始めました。音楽理論の「難しそう」という第一印象を取り払い、その本当の面白さを誰もが自分のペースで発見できるようにするための RAG システムです。

---

## 何をするか

- **質問応答**: 「ドミナントモーションとは?」のような質問に、教材を根拠に日本語で解説します
- **根拠の提示**: 解説の出典となった教材チャンクと類似度スコアを返します
- **音響解析**: アップロードされた音源から BPM・キー・コード進行を抽出し（librosa + BTC-ISMIR19）、理論解説に結びつけます

---

## 現状（MVP）

- **コーパス**: 2 系統。
  - フル版（ローカル）: SoundQuest の一般公開記事 162 本（Qdrant `music_theory_structure` に 2,355 points、構造ベース chunking）。会員限定記事 29 本は権利配慮のため除外（`scripts/check_gated.py` で検出）。
  - 公開デモ版: オープンライセンス教材コーパス（Qdrant `music_theory_open`）。権利上共有できるため公開デモの検索対象。
- **検索・生成**: 質問 → embed → search → generate の E2E が動作します。Streamlit UI（`apps/streamlit_app.py`）から利用でき、公開デモは Hugging Face Spaces で稼働します。
- **音響解析**: 音源のテンポ・キー・コード進行を解析し（BTC-ISMIR19、フォールバックはテンプレートマッチング）、解析結果を根拠に加えた解説を生成します。
- **評価基盤**: hit-rate@k / MRR（常用）と RAGAS 5 指標（節目のみ）の 2 層評価。20 問の Q&A セットで chunking 戦略の A/B 比較を実施済みです（下記）。

---

## 評価と改善の記録

検索層は **hit-rate@k / MRR（LLM 不使用・常時実行可能）** と **RAGAS（LLM judge・節目のみ）** の 2 層で評価しています。
生成層と評価 judge は、self-preference bias を避けるため意図的に別プロバイダにしています
（現行: 生成 = NVIDIA `meta/llama-3.3-70b-instruct`、judge = Gemini `gemini-3.1-flash-lite`）。
なお下表の A/B 実測（2026-07）は当時の生成層 `gemini-3.5-flash` で取得した記録です（[docs/evaluation.md](docs/evaluation.md)）。

### chunking 戦略の A/B 比較（n=20、2026-07）

固定長 800 字 chunking と、見出し境界で分割し breadcrumb 文脈を付与する構造ベース chunking を比較しました。

| 指標 | fixed | structure | diff |
| --- | --- | --- | --- |
| hit-rate@5 | 0.85 | 0.85 | ±0 |
| MRR | 0.792 | 0.800 | +0.008 |
| **context&#95;precision** (RAGAS) | 0.653 | **0.788** | **+0.134** |
| context_recall (RAGAS) | 0.667 | 0.717 | +0.050 |
| faithfulness / answer_relevancy / answer_correctness | — | — | 横ばい |

- **hit-rate では両者の差がゼロ**でしたが、RAGAS の context_precision で明確な差が可視化されました。「正解記事が top-k に入ったか」だけでは、取得チャンクの中身の質（ノイズ混入）は測れないためです。
- 固定長 chunking の弱点だった比較形式の質問（例:「3m はトニックかサブドミナントか」）が context_precision/recall 0.0 → 1.0 に回復しました。
- この結果を受けて**構造ベース chunking を本番採用**しています（`CHUNK_STRATEGY` で切替可能）。
- **残課題**: 度数表記の比較質問（「5-1 と 4-1 の違い」等）は構造 chunking でも 0.0 のままです。表記ゆれと複数記事にまたがる比較は chunking では解決できず、ハイブリッド検索・クエリ拡張の動機になっています。answer_correctness の低さ（≈0.37）も粒度ミスマッチ疑いとして未解決です。

詳細な per-question 比較は [docs/evaluation.md](docs/evaluation.md) を参照してください。

---

## アーキテクチャ

> **公開範囲について**: 取り込み（ingest）パイプラインの実装コード（`ingest.py` / `main.py` /
> scrape・ingest スクリプト）は、SoundQuest コーパスの権利配慮のため**このリポジトリには含めていません**。
> 公開しているのは、構築済みの Qdrant から検索・生成する本番経路（query）と評価基盤です。
> 以下の取り込みパイプラインの説明は設計の記録であり、対応するコードは非公開である点にご留意ください。

ユーザーが直接利用する UI は Streamlit（`apps/streamlit_app.py`）です。FastAPI + Inngest は、取り込みパイプライン（`rag_ingest`）が
バックグラウンドジョブとして正しく動作するかを検証するための実験用構成として用意しています。Inngest のジョブは
HTTP 経由で公開する必要があり、Inngest の Python SDK が公式に提供する FastAPI 用アダプタ（`inngest.fast_api.serve`）を
利用する形で、`music_rag/main.py` に最小限の FastAPI アプリを立てています。

取り込み（ingest）と検索・生成（query）で実行モデルを分けています。

- **ingest は Inngest の非同期ジョブ**: 162 記事の埋め込みは BGE-M3（約2GB）を伴う重い処理で、
  途中失敗からの再開や同時実行数の制御が必要です。`main.py` の `rag_ingest` が
  `scrape → chunk → embed → upsert` を Inngest の `step` として実行し、`concurrency=1` で
  メモリ枯渇を防いでいます。
- **query は同期パイプライン**: UI は質問に対してその場で回答を返す同期性が必要なため、
  `rag_query` と同じ流れ（embed → search →(audio)→ generate）を Inngest を介さない
  同期関数（`query_pipeline.py`）として実装しています。Streamlit UI はこれを直接呼び出します。

```mermaid
flowchart TD
    subgraph entry["入口"]
        ST[Streamlit<br/>apps/streamlit_app.py] --> QP[query_pipeline.py<br/>同期]
        FA[FastAPI app] --> ING[inngest serve<br/>非同期]
    end

    subgraph orch["オーケストレーション層 (main.py)"]
        RI[rag_ingest function]
        RQ[rag_query function]
    end

    subgraph proc["処理モジュール (pure functions)"]
        ingest[ingest.py<br/>scrape / chunk]
        embedder[embedder.py<br/>BGE-M3]
        retriever[retriever.py<br/>Qdrant]
        audio[audio.py<br/>librosa + BTC-ISMIR19]
        llm[llm.py<br/>NVIDIA Build]
    end

    subgraph ext["外部"]
        SQ[(SoundQuest)]
        QD[(Qdrant)]
        GM[NVIDIA Build API]
    end

    ING --> RI
    ING --> RQ

    RI -->|scrape| ingest --> SQ
    RI -->|chunk| ingest
    RI -->|embed| embedder
    RI -->|upsert| retriever --> QD

    QP -->|embed-query| embedder
    QP -->|search| retriever --> QD
    QP -->|analyze-audio| audio
    QP -->|generate| llm --> GM

    config[config.py] -.設定.-> proc
    types[custom_types.py] -.型.-> orch
```

---

## 設計方針（レイヤリング）

- **処理モジュールは純粋に保つ**: `ingest` / `embedder` / `retriever` / `llm` / `audio` は
  Inngest も `custom_types` も import しません。入出力は素の `dict` / プリミティブです。
- **オーケストレーション（接着剤）は2つ**: Inngest 経路（`main.py`）と同期経路（`query_pipeline.py`）です。
  どちらもモジュール間のインターフェース不一致を吸収します（例: `retriever` の出力
  `{"text","source","score"}` → `llm` が期待する `{"text","meta":{"source":...}}` への詰め替え）。
- **`custom_types` は step 境界専用**: Inngest の `step` が出力を JSON シリアライズする箇所の
  型検証にのみ使用します。同期経路（`query_pipeline.py` / `app.py`）には step 境界がないため使いません。
- **依存方向**: `custom_types ← main.py → ingest / retriever / llm`。`main.py` だけが両方を知ります。
- **冪等性**: Qdrant の point ID は `source + chunk_index` から決定的に生成され、再投入で上書きされます。

### モジュールインターフェース契約

```text
embedder.embed_query(str)            -> list[float]            # 1024 次元
embedder.embed_documents(list[str])  -> list[list[float]]
retriever.upsert(chunks, vectors)    -> {"ingested": int, "source": str}
retriever.search(vector, top_k)      -> [{"text","source","score"}, ...]
```

---

## ディレクトリ構成

本番運用コード（`src/music_rag/`）と、UI・運用 CLI・実験を役割別に分離しています。

```
├── src/music_rag/            # 本番パイプライン（pip パッケージ）
│   ├── main.py               #   オーケストレーション層（Inngest function + FastAPI 入口）
│   ├── query_pipeline.py     #   同期版クエリパイプライン（UI から直接呼ぶ）
│   ├── config.py             #   設定の一元管理（collection・モデル・chunking 戦略・データパス）
│   ├── custom_types.py       #   Inngest step 境界の Pydantic 型（main.py のみが使用）
│   ├── ingest.py             #   スクレイプ + チャンク分割（fixed / structure の 2 戦略）
│   ├── embedder.py           #   BGE-M3 埋め込み（dense 1024 次元）
│   ├── retriever.py          #   Qdrant upsert / search
│   ├── llm.py                #   LLM による解説生成（NVIDIA Build / OpenAI互換API）
│   ├── audio.py              #   音響解析（テンポ・キー・コード進行）
│   ├── audio_source.py       #   音声URL入力の解決レイヤー（YouTube / ニコニコ → 一時ファイル）
│   └── model/                #   BTC-ISMIR19 vendoring（コード認識モデル）
├── apps/
│   └── streamlit_app.py      # Streamlit UI（質問・回答・出典・音響解析結果を表示）
├── scripts/                  # 運用 CLI
│   ├── scrape_all.py         #   全記事を一括スクレイプして data/raw/ に保存
│   ├── ingest_all.py         #   clean 記事の取り込みを Inngest に fan-out
│   └── check_gated.py        #   会員限定記事を検出し data/reports/ にレポート出力
├── experiments/
│   └── evaluation.py         # hit-rate@k / MRR / RAGAS、chunking 戦略の A/B 比較
├── docs/                     # デモ画像・評価結果詳細
└── data/                     # コーパス・評価データ（権利保護のため git 管理外）
```

---

## 技術スタック

- **UI**: Streamlit（公開デモは Hugging Face Spaces / Streamlit SDK）
- **取り込みパイプラインの動作検証用**: FastAPI + Inngest
- **ベクトルDB**: Qdrant（ローカルは Docker、公開デモは Qdrant Cloud。cosine, 1024 次元）
- **埋め込み**: BGE-M3（dense 1024 次元。FlagEmbedding で自前実行。
  将来 sparse/hybrid に拡張可能）
- **生成**: NVIDIA Build（`meta/llama-3.3-70b-instruct`、OpenAI互換API）
- **評価**: hit-rate@k / MRR（自作）+ RAGAS 0.4（judge は Gemini `gemini-3.1-flash-lite` に分離）
- **音響解析**: librosa（テンポ・キー）+ BTC-ISMIR19（コード認識、large_voca）
- **言語/環境**: Python 3.11（conda + uv、src layout パッケージ）

---

## セットアップ

前提: Docker, Python 3.11+, conda, uv
（URL入力機能を使う場合は ffmpeg も必要: `brew install ffmpeg`。動かなくなったらまず `uv lock --upgrade-package yt-dlp` で yt-dlp を更新）

```bash
# 1) Python 環境（music_rag パッケージが editable install される）
conda activate rag-music-theory
uv sync

# 2) Qdrant（Docker）を起動
docker compose up -d

# 3) .env を作成（雛形: .env.example）
cp .env.example .env
#   最低限:
#   NVIDIA_API_KEY   … 生成層（必須）
#   GEMINI_API_KEY   … RAGAS 評価を回す場合のみ（judge 用）
#   任意:
#   QDRANT_CLOUD_URL / QDRANT_CLOUD_API_KEY      … Cloud へ collection を転送する場合
```

> コマンドはすべてリポジトリルートから実行してください（データパスは `./data` 基準です。
> 別の場所から実行する場合は環境変数 `MUSIC_RAG_DATA_DIR` で上書きできます）。

> **教材コーパスについて**: 著作権の都合により、コーパス本体（`data/`）はリポジトリに含めていません。
> コードとアーキテクチャは閲覧できますが、動作には別途コーパスの取り込みが必要です。
> 動作の様子は上記 Demo をご覧ください。

## 使い方

```bash
# 取り込み（SoundQuest にアクセスするのはこの段階のみ）
uv run python scripts/scrape_all.py    # 未取得分だけローカル保存
uv run python scripts/check_gated.py   # 会員限定記事を検出（data/reports/ にレポート）
uv run python scripts/ingest_all.py    # chunk → embed → upsert を Inngest に fan-out
#   ※ Inngest 経路を使う場合は別ターミナルで:
#   uv run uvicorn music_rag.main:app --reload
#   npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest

# デモ UI（質問 → 検索 → 生成。音声ファイルのアップロード / YouTube・ニコニコURL入力にも対応）
uv run streamlit run apps/streamlit_app.py

# 検索品質の評価
uv run python experiments/evaluation.py             # hit-rate@k / MRR（全戦略比較）
CHUNK_STRATEGY=fixed uv run streamlit run apps/streamlit_app.py   # 旧chunkingとのA/B
```

---

## ロードマップ

実施済み:

- ~~チャンク品質の刷新~~: 固定長分割 → 構造ベース分割。RAGAS で context_precision +0.134 を確認し本番採用（上記「評価と改善の記録」）
- ~~生成品質の評価~~: RAGAS 5 指標評価を chunking A/B の節目で実施
- ~~音声入力~~: アップロード音源の解析（librosa + BTC-ISMIR19）と解説生成への接続
- ~~音声URL入力~~: YouTube / ニコニコ動画URLからの解析（yt-dlp。ローカル個人利用限定の機能で、公開デプロイ時は `ENABLE_URL_INPUT=false` で無効化）
- ~~公開デプロイ~~: Hugging Face Spaces（Streamlit SDK, 無料CPU）へオープンライセンスコーパス版を公開。生成は NVIDIA Build、検索は Qdrant Cloud、埋め込み・音響解析は Space 内で実行。`main` への push で GitHub Actions が自動反映。
- ~~生成層のプロバイダ移行~~: Gemini 無料枠の RPD 枯渇を避けるため生成を NVIDIA Build に移行。RAGAS judge は bias 回避のため Gemini 側に分離。

今後:

- **hybrid / sparse 検索**: BGE-M3 のフラグ切り替えで sparse ベクトルを有効化します。度数表記の比較質問（評価で残った弱点）への対策です。
- **LLM モデルの柔軟性**: ユーザーが好みのモデルを選択できるようにします（生成層は既に OpenAI 互換API化済みで下地はあります）。
- **メロディ解析（F0）・セグメント分割**: 音響解析の拡張です。
- **SoundQuest 版の公開**: 権利者の許諾確認後に、フル版（SoundQuest コーパス）の公開可否を判断します。

---

## Credits / Third-Party

- [src/music_rag/model/](src/music_rag/model/) 配下のコード認識機能は [BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19)
  （Jonggwon Park, "A Bi-Directional Transformer for Musical Chord Recognition", ISMIR 2019）
  の一部をvendoringしています（MIT License, Copyright (c) 2019 Jonggwon Park）。
  ライセンス全文は [src/music_rag/model/LICENSE_BTC-ISMIR19](src/music_rag/model/LICENSE_BTC-ISMIR19) を参照してください。
