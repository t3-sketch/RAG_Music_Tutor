# CLAUDE.md — music-rag (RAG_Music_Tutor)

このドキュメントは、Claude（および他のAIアシスタント/コラボレーター）がこのリポジトリで作業する際のオンボーディング資料。プロジェクトの目的・アーキテクチャ・設計判断・作業規約をまとめる。

最終更新: 2026-07-19（Phase 5: 評価セット66問化・検索層ベースライン再計測完了 / RAGAS未実施）

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
| --- | --- | --- |
| Embeddings | BGE-M3 | dense 1024次元。FlagEmbeddingでローカル実行のみ（リモートバックエンドは2026-07-13に全廃。DeepInfra経路は一度も使われず、NVIDIA Build版bge-m3はサーバー側500で使用不可のため）。ハイブリッド検索は将来課題 |
| Vector DB | Qdrant | ローカルは Docker（bind mount: `data/qdrant/`）、公開デモは Qdrant Cloud |
| 生成層 LLM | **Gemini `gemini-3.5-flash`**（既定） / NVIDIA Build (`meta/llama-3.3-70b-instruct`) | `llm.py` 経由（両者ともOpenAI互換API）。**`config.LLM_PROVIDER`（env `LLM_PROVIDER=gemini\|nvidia`）で切替**。NVIDIAは応答が遅くUIが待たされるため既定をGeminiに（2026-07-16）。**評価バッチ（RAGAS）は `LLM_PROVIDER=nvidia` を推奨** — GeminiはRPD上限が厳しく、judge(Gemini)とRPDを食い合うため。Claude APIはMVPでは不使用 |
| 評価 | RAGAS 0.4.3 | judge: Gemini `gemini-3.1-flash-lite`（`config.GEMINI_MODEL`。生成層とは別プロバイダに分離しbias回避）。langchain-community==0.3.27 ピン留め必須 |
| デプロイ | Hugging Face Spaces (Streamlit SDK, 無料CPU) | `main` push で GitHub Actions が同期。オープンコーパス版(`music_theory_open`)を公開。URL入力はオフ |
| ジョブオーケストレーション | Inngest (v0.5.18) + FastAPI | FastAPIはInngestアダプター層 |
| UI | Streamlit (`apps/streamlit_app.py`) |  |
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
- 例: 構造ベースchunking実験は `chunk_structure()` + `music_theory_structure` collection で行い、当時の本番 `music_theory` collection は不変に保った（その後 structure を本番昇格。§4参照）

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
| --- | --- |
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

### Phase 4: 音声URL入力（YouTube / ニコニコ）— 完了（2026-07-05）
- **`audio_source.py` 新設**: URL → ローカル一時ファイルパスの解決レイヤー。「下流はすべて audio_path を話す」契約の手前に置き、`audio.py` は無変更。ローカルパスは素通し。
  - 許可ホストallowlist（YouTube / ニコニコのみ。SSRF的悪用の防止兼用）
  - ダウンロード前にメタデータのみ取得し `MAX_AUDIO_DURATION_SEC`（default 600秒）で弾く
  - `ENABLE_URL_INPUT` ゲートは resolve() 自身が enforcement。**利用規約の関係でローカル個人利用限定・公開デプロイ時は false**
  - yt-dlpエラー → ユーザー向け日本語メッセージ変換（非公開/削除済/年齢制限/会員限定/地域制限）
  - フォーマットは mp3 mono 採用（audioreadフォールバック警告の回避 + サイズ半減。`change.md` 2026-07-05 に計測表）
- **検証完了（2026-07-05）**: ①UIからの完全E2E（YouTube URL → yt-dlp → BTC解析 128BPM/D#maj/vi-IV-I-V → Gemini生成・出典表示）、②ニコニコ実URL（sm9・nico.ms 短縮URL、削除済み動画のエラー変換含む）、③Inngest経由（`rag/query` イベントに audio_path=URL、step内 resolve→analyze→cleanup 完走・一時dir残骸なし）。

### Phase 5: 評価基盤の刷新（進行中・2026-07-13〜）

**ゴール:** 「検索改善（ハイブリッド検索/クエリ拡張）で本当に良くなったか」を統計的に主張できる評価基盤を作る。そのために (a) 評価セットを統計的検出力のある規模（silver 20 → 100問超）に拡張し、(b) 多ソース比較質問を測れる指標に刷新し、(c) A/B判断を集計平均でなく paired difference で行う。

- **実験設計ガイドを整備**: `docs/experiment-design.md`。Web調査（Anthropic "Adding Error Bars to Evals" 2024、Google Cloud、Nirant Kasliwal 等）を現状パイプラインに引きつけて整理。核心の知見は「hit-rate 0.85・n=20 の 95%CI は ±0.16 → n=20〜33 では戦略差はほぼ検出不能」。
- **forum由来の評価データセット（`data/eval/forum_review.json`, gitignore）**: SoundQuestフォーラムの実質問129件。正解ソースは**複数記事対応（list）**で、比較/統合質問を測れる。各問に `match_type`（single/and/or）を持つ。
  - レビュー完了（2026-07-16）: 優先バッチ24件（flags付き+confidence low/med）を NotebookLM+人手で検証（ADOPT 13 / EXCLUDE 11）。残り80件（全て confidence high）を **NotebookLM API 経由で検証**（`experiments/apply_nblm_review.py`）→ **ADOPT 27 / EXCLUDE 52 / 保留 1**。
  - **重要な発見: 元パイプラインの `answerable_standalone=true` ラベルは楽観的すぎた。** 除外率65%で、実体はサイトへの機能要望・誤字報告・学習相談・自作曲の分析依頼など「記事では原理的に答えられない投稿」。フォーラムを評価セット化する際は answerable 判定こそが本体の作業。
  - **NotebookLM運用の知見**: ワークシートを**ソースとして**読ませると内容を部分的にしか拾えず「質問文が省略されていて読めない」と誤EXCLUDEする（10問中5問で発生）。**質問文をプロンプトに直接インライン**すれば解消（記事側の読み取りは正常）。「読めなかった」起因のEXCLUDEは判定として無効なので、`apply_nblm_review.py` が正規表現で検出して保留に回す。実在しないslugも `data/raw` 照合で自動排除。
- **統合eval set `experiments/build_eval_set.py` → `data/eval/eval_set_merged.json`**: silver 20 + forum(reviewed) を統一スキーマにまとめる。各問に `reviewed` フラグ（未検証ラベルを必ず区別）。`--include-pending` で未レビュー分をタグ付き投入も可。現状 **66問**（silver 20 + forum reviewed 40 + generated 6、うち AND 27 / OR 19 / single 20）。generated 6 は度数→コード足場の検証用（キー明示の度数質問）。
- **指標を多ソース対応に刷新（`experiments/evaluation.py`）**: 主指標を **recall@k**（top-kに入った正解記事の割合・連続値）に。加えて **strict hit-rate**（and=全記事必須/or・single=1つでOK）と MRR。source/match_type/difficulty で**層別集計**（全体平均が比較質問の弱点を隠すため）。単一ソース経路は不変で、silver 20問が過去値 0.85/0.792 を完全再現（回帰なし確認）。
- **paired difference 分析（`experiments/paired_diff.py`）**: 保存済み per-question から fixed vs structure を突き合わせ、平均差の bootstrap 95%CI と Wilcoxon 検定を出す。
- **検索層スコア（66問, 2026-07-19, `data/eval/scores_20260719.json`）**:

  | | recall@5 | strict_hit | MRR |
  | --- | --- | --- | --- |
  | fixed | 0.571 | 0.470 | 0.581 |
  | structure | 0.599 | 0.500 | 0.562 |

- **AND質問（多ソース比較・統合, n=27）は依然壊滅的**: structure recall 0.352 / strict_hit 0.111。n=9→27 に増えても改善せず、**偶然では説明できない確かな弱点**として定量化できた（単一ソースhit-rateでは見えない）。→ ハイブリッド検索/クエリ拡張の動機が確定。
- **paired diff の結論（fixed→structure, n=60時点。66問での再計算は未実施）**: 3指標すべて有意差なし。recall +0.031（CI[−0.053,+0.114], p=0.58）、strict_hit +0.033（p=0.69）、**MRRは符号反転 −0.023**（改善9/悪化10, p=0.41）。60問中49問が同一。**n=33でも n=60でも結論は変わらず、retrieval指標で structure 優位は主張できない。** silver 20問は fixed/structure で1問も動かない（retrieval指標はもともと両者を区別しない。structureの価値は過去のRAGAS context_precision +0.134 でのみ観測されている）。
- **当初「100問で検定力確保」の見込みは外れた**: answerable が実際には少なく（除外52件）採れたのは66問（forum由来40 + silver 20 + generated 6）。かつ retrieval 指標自体が両chunking戦略を区別しないため、**問題は n ではなく指標の選択**の可能性が高い。structure の評価は RAGAS context_precision で行うべき。

### 次にやること（Phase 5 続き）
1. **RAGAS（生成層）を66問で実行** — 未実施。生成=Gemini flash-lite / judge=MiniMax M3（OpenRouter, `JUDGE_PROVIDER=openrouter`）にプロバイダ分離済み（2026-07-19、.env）。まず **context_precision の paired diff**（structureの真価が出る指標・コール数最小）から。
2. 度数表記の比較質問対策（ハイブリッド検索 / クエリ拡張 / メタデータ）← AND recall 0.352 (n=27) という確かな動機
3. 保留1件 + 未レビュー26件（answerable=false 判定済み含む）の扱いは必要になったら

### 既知の未解決事項
- **「5-1と4-1の違い」「7-1と4-3の解決の違い」（度数表記の比較質問）は構造chunkingでも context&#95;precision/recall = 0.0** → chunkingでは解けない検索課題（表記ゆれ・複数記事にまたがる比較）。ハイブリッド検索/クエリ拡張の動機。
- **answer&#95;correctness は依然低い（0.37）** → 粒度ミスマッチ疑い。生成層・評価セット側の課題。
- ~~構造chunkingを本番採用するか~~ → **検索先としての採用は完了済み**（refactor Phase 2 で `CHUNK_STRATEGY` の default が `"structure"` になり、`COLLECTION_NAME` = `music_theory_structure` が本番の検索先。`CHUNK_STRATEGY=fixed` でA/B用に旧経路へ切替可）。
- **旧 `music_theory` collection（固定長chunking・1,502 points）をQdrantから削除するかは未決定** → TM判断待ち（残す場合はA/B比較用という位置づけ）。
- ~~音声つき完全E2E（UI→生成）は未実施~~ → **完了（2026-07-05）**。UI・Inngest両経路でURL入力込みのE2Eを実走確認（Phase 4 参照）。

---

## 5. 環境注意事項

- librosa追加時、numbaの制約で numpy が 2.5.0 → 2.4.6 に自動ダウングレードされた
- torch 2.12.1 導入済み
- **librosa 0.10+ の破壊的変更:** `beat_track` 等のtempo戻り値が配列化 → `float(np.atleast_1d(tempo)[0])` で対応（修正済み）
- **thinkingモデルに `max_tokens` を引き継ぐと回答が消える（2026-07-16）:** `max_tokens` は「そのレスポンスの総生成量」の上限で、thinkingモデル（`gemini-3.5-flash`）では**思考トークンと可視回答が同じ予算を食い合う**。非thinking（NVIDIA llama-3.3）向けに調整した `MAX_TOKENS=1500` をそのまま使うと、思考が~1430tok消費し**可視出力66tokで `finish_reason=length`** → 回答が「テンポは約 **」で途切れる。対処: `llm.py` の生成呼び出しでは **max_tokens を指定しない**（モデル既定の大きい上限に委ねる）。
- **生成API呼び出しには必ず明示 timeout を（2026-07-16）:** OpenAI SDK既定は約600秒。NVIDIA側が遅いとUIが10分ハングする（実際に発生）。`config.LLM_TIMEOUT_SEC`（既定90秒）を client に渡している。`llm.py` の tenacity リトライは `InternalServerError`(5xx) にしか発火しないので、「ただ遅い」ケースはリトライされず待ち続ける点に注意。
- RAGAS × Gemini統合の落とし穴:
  - instructorアダプタ・litellmアダプタともに `google-genai` ネイティブクライアントの非同期判定に失敗
  - 解決策: GeminiのOpenAI互換エンドポイント経由で `AsyncOpenAI` + `provider="openai"`
  - `max_tokens=8192` 必須（未指定だと日本語の複数statement照合で `IncompleteOutputException`）

---

## 6. スコープと将来課題

### コード認識スコープ
- MVPは MajMin + 7th まで
- テンションコード（9th/11th/13th）は将来課題。音源分離が前提条件で、学術的にも未解決に近い領域（BTC-FDAA-FGF等の論文で確認済み）

### 次のマイルストーン（Phase 5 進行中 → §4 Phase 5 参照）
- **forum残り80件のレビュー → eval set 100問超 → 検定力確保**（最優先。これが無いと以下の効果を統計的に主張できない）
- 度数表記の比較質問対策（ハイブリッド検索 / クエリ拡張 / メタデータ）← AND質問 recall 0.26 という定量的動機がついた
- 旧 `music_theory` collection（fixed chunking）を削除するかの判断（A/B比較の検定力が付くまで残置が無難）
- 音声解析結果によるretrievalクエリ拡張（現状は生成層にのみ寄与）

### バックログ
- ~~YouTubeリンクからの音声解析~~ → Phase 4 で完了（YouTube / ニコニコURL入力。§4参照）
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
| --- | --- | --- |
| `data/raw/*.json` | SoundQuest 162記事 | gitignore（権利保護のためローカルのみ） |
| `data/eval/` | 20問silver Q&A、RAGAS結果 | gitignore（ローカルのみ） |
| `data/qdrant/` | Qdrant bind mount | gitignore |
| `model/` | BTC-ISMIR19 vendoring（MIT、帰属表記必須） | コミット対象 |

**注意:** corpusおよび評価データは絶対にリポジトリにコミットしない。パブリックデプロイは権利者許諾が取れるまで行わない。
