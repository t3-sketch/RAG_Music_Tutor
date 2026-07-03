# CLAUDE.md — music-rag (RAG_Music_Tutor)

このドキュメントは、Claude（および他のAIアシスタント/コラボレーター）がこのリポジトリで作業する際のオンボーディング資料。プロジェクトの目的・アーキテクチャ・設計判断・作業規約をまとめる。

最終更新: 2026-07-02（Phase 3 完了時点）

---

## 1. プロジェクト概要

### 目的
SoundQuest（soundquest.jp、作者: 紅雪）の音楽理論記事をcorpusとした日本語RAGシステム。

**最終ゴール:** 音声ファイルを受け取り、コード進行・メロディを解析し、SoundQuestの音楽理論に基づく説明を生成すること。

### 背景・位置づけ
- 日本語サブカルチャー音楽ジャンル（同人音楽・Kawaii Future Bass等）のディスカバリー改善を目指す「Music Controller」構想の基盤
- ポートフォリオ兼インターン応募用プロジェクト
- **パブリックデプロイは権利者（紅雪氏）への許諾確認が取れるまで保留**

### Corpus
- SoundQuest 162記事（メンバー限定29記事は除外）
- `data/raw/*.json` に格納（gitignore済み、ローカルのみ。権利上リポジトリには含めない）

---

## 2. 技術スタック

| レイヤー | 技術 | 備考 |
|---|---|---|
| Embeddings | BGE-M3 (FlagEmbedding) | dense 1024次元。ハイブリッド検索（sparse+dense）は将来課題 |
| Vector DB | Qdrant | Docker、bind mount: `data/qdrant/` |
| 生成層 LLM | Gemini (`gemini-3.5-flash`) | `llm.py` 経由（`config.GEMINI_MODEL`、env `GEMINI_MODEL` で上書き可）。Claude APIはMVPでは不使用 |
| 評価 | RAGAS 0.4.3 | judge: `gemini-3.1-flash-lite`。langchain-community==0.3.27 ピン留め必須 |
| ジョブオーケストレーション | Inngest (v0.5.18) + FastAPI | FastAPIはInngestアダプター層 |
| UI | Streamlit (`apps/streamlit_app.py`) | |
| 音響解析 | librosa + BTC-ISMIR19 (large_voca) | `src/music_rag/model/` 配下にvendoring済み（MIT） |
| Python環境 | conda (Python 3.11) + uv | condaはinterpreter管理、uvはパッケージ管理。src layout・editable install |

作業パス: `/Users/macuser/dev/RAG_Music_Theory`

### ディレクトリ構成（2026-07-02 リファクタ後）
- `src/music_rag/` — 本番パイプライン（pipパッケージ。config / main / query_pipeline / ingest / embedder / retriever / llm / audio / custom_types / model）
- `apps/streamlit_app.py` — UI入口
- `scripts/` — 運用CLI（scrape_all / ingest_all / check_gated）
- `experiments/` — evaluation.py（hit-rate/MRR/RAGAS、chunking A/B）
- `docs/` — demo.png、evaluation.md（評価数値の公開用置き場）

### 起動コマンド（リファクタ後）
```bash
uv run streamlit run apps/streamlit_app.py
uv run uvicorn music_rag.main:app --reload      # 旧: uvicorn main:app
uv run python scripts/ingest_all.py
uv run python experiments/evaluation.py
```
**コマンドはrepoルートから実行する規約**（データパスは `config.DATA_DIR`＝cwd基準の `./data`。`MUSIC_RAG_DATA_DIR` で上書き可）。

---

## 3. アーキテクチャ原則

### モジュール境界はprimitive型で
- processing modules（`ingest`, `embedder`, `retriever`, `llm`, `audio`）は `custom_types` をimportしない
- 型変換の責任は `main.py` が持つ

### 生成層と評価層は意図的に別モデル
- 同一モデルだとself-preference biasでjudgeが甘くなる
- ポートフォリオで説明できる設計判断として重要

### 実験的変更は本番から完全に分離
- 新関数 + 新collectionで二重隔離
- 例: 構造ベースchunking実験は `chunk_structure()` + `music_theory_structure` collection で行い、本番の `music_theory` collection は不変に保つ

### upsertのidempotency
- point IDは `uuid5(source:chunk_index)` でdeterministic
- upsertは冪等 → 再実行前にcollectionをdropする必要はない

---

## 4. 現在の状態

### Phase 1: 検索・生成層 — 完了
- Qdrant `music_theory` collection: 1,502 points（固定長chunking）
- `query_pipeline.py`（同期グルーモジュール）・`app.py`（Streamlit UI）実装済み、E2Eデモ動作確認済み
- RAGAS評価パイプライン構築・全20問実行完了
  - 結果: `data/eval/ragas_music_theory_20260701.json/.csv`（gitignore済み、ローカルのみ）

**RAGAS最終スコア:**

| Metric | Score |
|---|---|
| faithfulness | 0.804 |
| answer_relevancy | 0.881 |
| answer_correctness | 0.361 |
| context_precision | 0.654 |
| context_recall | 0.667 |

**評価の知見:**
- 生成層は健全、検索層に改善余地
- 比較/選択形式の質問で context_precision/recall = 0.0（固定長chunkingの限界。hit-rateでは見えなかった弱点）
- answer_correctnessの低さは粒度ミスマッチの疑い

### Phase 2: 音響解析 — 完了（2026-07-01）
- **Songle API依存を撤廃**（ニッチジャンルはSongle DB未収録の可能性が高いため）、自前実装/HFモデル路線へ変更
- BTC-ISMIR19（large_vocaモデル）によるコード認識を実装
  - vendoringファイル: `model/btc_model.py`, `transformer_modules.py`, `btc_chords.py`, `btc_infer.py`, `btc_large_voca.pt`
  - ライセンス表記完備: `model/LICENSE_BTC-ISMIR19`、各ファイル帰属コメント、README Creditsセクション
- `audio.py` 設計: **案B確定（chroma特徴量共有アーキテクチャ）**
  - 内部関数 `_load_and_chroma(path)` → `detect_key(chroma)` と `detect_chords()` で共有
  - `detect_tempo_beats(path)`, `detect_key(chroma)`, `detect_chords(path, ...)`（BTC推論 → テンプレートマッチングfallback）
  - `analyze(path)`: tempo/beats/key/chordsをフラットな時系列dictで返す（将来のsegment分割を後付けしやすい設計）
- 実機検証済み: BTC約17.8秒/236区間、fallback 204区間、音楽的に整合するコード出力を確認

### Phase 3: パイプライン接続 — 完了（2026-07-02）
- **main.py のSongle参照を修正**: `rag_query` step3 が `audio.analyze()` + `audio.describe()`（新設）を呼ぶ形に。`QueryEventData.songle_url` → `audio_path` に変更。
- **audio.py と RAGパイプラインの統合**: `query_pipeline.answer_query(audio_path=...)`、`app.py` は音声ファイルアップローダー（一時ファイル経由）に。audio_desc は生成層プロンプトにのみ渡る（検索には影響しない。検索クエリ拡張は将来の設計判断）。
- **BGE-M3 × numpy 2.4.6 動作検証済み**（問題なし。hit-rateスコアも6/30と完全一致）。
- **構造ベースchunking評価完了**（`music_theory_structure`: 162記事・2,355 points）:
  - hit-rate@5 / MRR: fixed 0.85/0.792 vs structure 0.85/0.80（差が見えない）
  - RAGAS: **context_precision 0.653→0.788 (+0.134)**、context_recall 0.667→0.717 (+0.050)、生成系3指標は横ばい。比較質問「3mってトニックかサブドミナントどっち」が cp/cr 0.0→1.0 に回復。
  - 結果: `data/eval/ragas_music_theory_structure_20260702.json/.csv`
- 作業ログの詳細は `change.md` を参照。

### 既知の未解決事項
- **「5-1と4-1の違い」「7-1と4-3の解決の違い」（度数表記の比較質問）は構造chunkingでも context_precision/recall = 0.0** → chunkingでは解けない検索課題（表記ゆれ・複数記事にまたがる比較）。ハイブリッド検索/クエリ拡張の動機。
- **answer_correctness は依然低い（0.37）** → 粒度ミスマッチ疑い。生成層・評価セット側の課題。
- **構造chunkingを本番採用するか（`music_theory` collection を置き換えるか）は未決定** → TM判断待ち。
- **音声つき完全E2E（UI→生成）は未実施**（gemini-3.5-flash free-tier日次quotaをRAGASで消費したため）。構成要素は個別検証済み。quota回復後にUIから一度通す。

---

## 5. 環境注意事項

- librosa追加時、numbaの制約で numpy が 2.5.0 → 2.4.6 に自動ダウングレードされた
- torch 2.12.1 導入済み
- **librosa 0.10+ の破壊的変更:** `beat_track` 等のtempo戻り値が配列化 → `float(np.atleast_1d(tempo)[0])` で対応（修正済み）
- RAGAS × Gemini統合の落とし穴:
  - instructorアダプタ・litellmアダプタともに `google-genai` ネイティブクライアントの非同期判定に失敗
  - 解決策: GeminiのOpenAI互換エンドポイント経由で `AsyncOpenAI` + `provider="openai"`
  - `max_tokens=8192` 必須（未指定だと日本語の複数statement照合で `IncompleteOutputException`）

---

## 6. スコープと将来課題

### コード認識スコープ
- MVPは MajMin + 7th まで
- テンションコード（9th/11th/13th）は将来課題。音源分離が前提条件で、学術的にも未解決に近い領域（BTC-FDAA-FGF等の論文で確認済み）

### 次のマイルストーン（Phase 4 候補・未確定）
- 構造chunkingの本番採用判断（採用なら `music_theory` の置き換え手順を設計）
- 度数表記の比較質問対策（ハイブリッド検索 / クエリ拡張 / メタデータ）
- 音声解析結果によるretrievalクエリ拡張（現状は生成層にのみ寄与）

### バックログ
- YouTubeリンクからの音声解析（現在の入力はローカル音声ファイルのみ）
- メロディ解析（F0）
- セグメント分割（Aメロ/Bメロ/サビ等の構造認識）
- テンションコード認識（音源分離が前提）
- 音源分離の再検討（テンション対応時に限り）
- 生成層LLMをユーザーが選べる機能（アダプターパターン）
- ハイブリッド検索（sparse + dense）
- リッチなチャンクメタデータ

---

## 7. 作業規約（Claudeへの指示）

- **アーキテクチャはTMが提案し、Claudeは確認・ヒントを出す**（直接答えを与えない）
- **根本原因の説明を重視**（表面的な修正より「なぜそうなるか」の理解を優先）
- **情報過多を避け、一度に一つの具体的事項を提示する**
- **変更は差分（diff）で提示する**（ファイル全体の書き直しは不要な変更を生むため避ける）
- **既存の動作中のコードは必要がない限り触らない**
- **大きなリファクタリングより、段階的で検証可能な変更を好む**
- **APIの挙動は必ずドキュメント/ソースで検証する**（自動補完・モデルの記憶を信頼しない。VSCode autocompleteとGeminiがInngestの存在しないAPIを提案して実バグを生んだ経験による）
- **会話は日本語、技術用語は英語のまま**

---

## 8. データとファイルの取り扱い

| パス | 内容 | Git |
|---|---|---|
| `data/raw/*.json` | SoundQuest 162記事 | gitignore（権利保護のためローカルのみ） |
| `data/eval/` | 20問silver Q&A、RAGAS結果 | gitignore（ローカルのみ） |
| `data/qdrant/` | Qdrant bind mount | gitignore |
| `model/` | BTC-ISMIR19 vendoring（MIT、帰属表記必須） | コミット対象 |

**注意:** corpusおよび評価データは絶対にリポジトリにコミットしない。パブリックデプロイは権利者許諾が取れるまで行わない。
