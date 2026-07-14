# RAG評価の実験設計ガイド

Web上のベストプラクティス調査（2026-07-13）を、本プロジェクトの評価パイプライン
（`experiments/evaluation.py`: hit-rate@5 / MRR + RAGAS 5指標、20問 silver set）に
引きつけてまとめたもの。**「どうすべきか」の最終判断はTMが行う**。本docは判断材料。

---

## 1. 実験設計の大原則

調査したすべてのソースが一致していた点:

1. **1実験1変数**（change one variable at a time）
   - chunking・embedding・top-k・プロンプト・生成モデルのうち、1回の実験で動かすのは1つだけ
   - 効いた設定は freeze してから次の変数へ。複数同時に変えると寄与が分離できない
   - → 本プロジェクトの「fixed vs structure を collection 分離で比較」はこの原則に合致
2. **retrieval層と生成層は独立に評価してから end-to-end**
   - ボトルネックの帰属（検索が悪いのか生成が悪いのか）を可能にするため
   - → hit-rate/MRR（LLM不使用・高頻度）と RAGAS（節目のみ）の2段構えは合致
3. **評価セット・パラメータは実験間で絶対に変えない**
   - 質問・ground truth・top-k・プロンプトが1文字でも変わったら、それは別の実験
   - 評価セットを更新したら**バージョン番号を付けて、過去スコアとの比較不能を明記**する
4. **本番ログからの評価セット継続更新**（本プロジェクトは未デプロイなので将来課題）

---

## 2. 評価データセット（golden dataset）の設計

### 推奨される規模と構成
- 手動キュレーションの golden set は **50〜100問** が実務上の推奨スタート
  （現状20問。§4の統計的検出力の観点でも20問は差を検出するには少ない）
- **質問タイプの多様性**を意図的に設計する:
  - simple（単一記事で答えられる事実質問）
  - comparison / multi-hop（複数記事にまたがる比較・推論）
  - 表記ゆれ・口語（「5-1」vs「V-I」vs「ドミナントモーション」）
  - 答えがコーパスに存在しない質問（無回答すべきケース）← 現状未カバー
- タイプごとの構成比を記録し、**タイプ別にスコアを層別報告**する
  （全体平均は comparison 質問の弱点を隠す — 実際 Phase 1 で hit-rate では見えず
  RAGAS context_precision の per-question 分析で初めて見えた）

### 拡張の選択肢
- **RAGAS の testset generation** でコーパスから合成質問を生成できる
  （reasoning / conditioning / multi-context のタイプ別比率を指定可能）。
  ただし合成データは多様性・現実性に欠ける傾向があるため、
  **合成→人手レビューで採否判断**が推奨フロー
- スコアが低かったサンプルは人間が見て「本当に悪いのか、評価セット側の不備か」を
  仕分けする（ground truth の間違いはよくある）

### 本プロジェクト固有の論点
- 現状の `expected_source` は**1問1記事**の前提。しかし未解決課題の
  「5-1と4-1の違い」型の比較質問は**正解ソースが複数記事**にまたがる。
  `expected_sources: list` にして **recall@k（複数正解のうち何割拾えたか）** を
  併記できる形にしないと、比較質問の検索改善を hit-rate では測れない

---

## 3. 指標の選び方（どの段階で何を見るか）

Nirant Kasliwal の stage 別ガイドが実務的に整理されている:

| 段階 | 指標 | 理由 |
| --- | --- | --- |
| 初期デバッグ | hit-rate@k | 「top-kに入ったか」の二値。安く速い |
| retriever チューニング | MRR + recall@k | 順位感度 + 網羅性 |
| reranker導入後 / システム評価 | nDCG@k + hit-rate | 複数正解の順位品質まで見る |
| 生成層 | faithfulness 最優先 | hallucination検出。RAGASでは最重要指標とされる |

- **現状の hit-rate@5 + MRR は「retrieverチューニング段階」として妥当**。
  nDCG は複数正解＋段階的関連度ラベルが前提なので、§2の multi-source 化を
  やるまでは導入コストに見合わない
- RAGAS 5指標の役割分担: context_precision/recall = 検索の質、
  faithfulness/answer_relevancy = 生成の質、answer_correctness = end-to-end。
  **answer_correctness は ground truth の粒度に強く依存**するため、
  低い値（0.37）を「システムが悪い」と読む前に評価セット側の粒度を疑うのは正しい筋

---

## 4. 統計的厳密性（n=20 で「差」を語れるか）

Anthropic「Adding Error Bars to Evals」(2024) の5推奨が最も体系的:

1. **平均だけでなく SEM（標準誤差）を報告する**: 95%CI = mean ± 1.96×SEM
2. **クラスタ標準誤差**: 同じ記事由来の質問が複数ある場合、質問は独立でないので
   naive な SEM は過小評価（最大3倍以上の差が出る）。記事単位でクラスタリング
3. **質問ごとに複数回生成して平均**する（LLM出力の分散を減らす）
4. **A/B比較は paired difference で行う**: 同じ質問セットを両条件に流し、
   **質問ごとの差分**の平均と CI を見る。質問難易度由来の分散が消えるので
   「タダで手に入る分散削減」
5. **power analysis** で「検出したい差」から必要な質問数を逆算する

### 本プロジェクトへの当てはめ（具体的な数字）

- hit-rate 0.85 / n=20 の SEM は √(0.85×0.15/20) ≈ **0.08 → 95%CI は ±0.16**。
  つまり「fixed 0.85 vs structure 0.85」どころか、**0.85 vs 0.70 ですら
  統計的には区別できない**。hit-rate で差が見えなかったのは chunking に差が
  ないからではなく、**n=20 では見える差がほぼ存在しない**から
- 一方 **paired 分析なら現状のデータで今すぐできる**: `scores_*.json` と
  `ragas_*.json` は per_question を保存済みなので、質問ごとの差分
  （structure − fixed）を並べて「何問で改善/悪化/不変か」を数え、
  符号検定や bootstrap CI をかけられる。集計平均の比較より遥かに感度が高い
- RAGAS context_precision +0.134 も、本来は paired difference の CI で
  「ゼロをまたがないか」を確認して初めて「改善した」と言える
- **LLM judge 自体が確率的**なので、RAGAS スコアの再現性（同一条件で2回流して
  どれだけブレるか）を一度測っておくと、以後「観測された差が judge ノイズか
  本物か」の判断基準になる

---

## 5. 推奨ワークフロー（調査結果の総合）

```
① baseline を確定（評価セット vN + 全設定を記録し freeze）
② 変更は1つだけ入れる
③ 同一評価セットで再実行（hit-rate/MRR は毎回、RAGAS は節目）
④ per-question の paired diff で差を見る（集計平均だけ見ない）
⑤ 質問タイプ別に層別して「どこが改善/悪化したか」を特定
⑥ 実験ログに記録（変更内容・仮説・結果・採否）→ 採用なら freeze、次の変数へ
```

- ①のために**実験条件のスナップショット**（CHUNK_STRATEGY, TOP_K, embedder,
  LLM モデル名, 評価セットのバージョン）をスコア JSON に含めると、
  後から「この数字は何の条件だったか」で悩まない（現状は collection 名と k のみ）
- ⑥は `change.md` の運用がすでに近い。CLAUDE.md グローバルルールの
  研究フレーム（課題/仮説/根拠/結果）をそのまま実験ログの様式にできる

---

## 6. 現状パイプラインとのギャップまとめ（TM判断用）

| # | ギャップ | 効果 | コスト |
| --- | --- | --- | --- |
| 1 | per-question paired diff 分析（既存データで可能） | A/B判断の感度が大幅向上 | 小（分析スクリプトのみ） |
| 2 | スコアJSONに実験条件スナップショットを埋める | 再現性・追跡性 | 小 |
| 3 | 評価セットに質問タイプのタグ付け + 層別集計 | 弱点の可視化 | 小〜中 |
| 4 | `expected_source` の複数化 + recall@k | 比較質問の検索改善を測定可能に | 中 |
| 5 | 評価セット拡張（20→50問以上、タイプ設計込み） | 統計的検出力 | 中〜大（人手レビュー前提） |
| 6 | RAGAS judge の再現性測定（同一条件2回実行） | 差の解釈基準 | 中（API消費） |
| 7 | 無回答ケース・nDCG 導入 | — | 4・5の後で検討 |

順序の目安: 1→2 は他の何をやるにも前提になる基盤。4 は未解決課題
（度数表記の比較質問）に直結。5 は「ハイブリッド検索で改善したか」を
主張したくなった時点で必須になる。

---

## Sources

- [Anthropic — A statistical approach to model evaluations](https://www.anthropic.com/research/statistical-approach-to-model-evals) / [arXiv: Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)
- [Google Cloud — RAG systems: Best practices to master evaluation](https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval)
- [Nirant Kasliwal — RAG Metrics for Technical Leaders](https://nirantk.com/writing/rag-metrics-for-technical-leaders/)
- [Anyscale Docs — RAG evaluation](https://docs.anyscale.com/rag/evaluation)
- [Microsoft Data Science — The path to a golden dataset](https://medium.com/data-science-at-microsoft/the-path-to-a-golden-dataset-or-how-to-evaluate-your-rag-045e23d1f13f)
- [Ragas Docs — Evaluation Dataset](https://docs.ragas.io/en/latest/concepts/components/eval_dataset/)
- [LangCopilot — RAG Evaluation 101: From Recall@K to Answer Faithfulness](https://langcopilot.com/posts/2025-09-17-rag-evaluation-101-from-recall-k-to-answer-faithfulness)
- [Orq.ai — Mastering RAG Evaluation 2026](https://orq.ai/blog/rag-evaluation) / [Maxim — Complete Guide to RAG Evaluation 2025](https://www.getmaxim.ai/articles/complete-guide-to-rag-evaluation-metrics-methods-and-best-practices-for-2025/)
