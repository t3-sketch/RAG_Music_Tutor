# RAG評価の実験設計ガイド

Web上のベストプラクティス調査（2026-07-13）を、本プロジェクトの評価パイプライン
（`experiments/evaluation.py`: 検索層 recall@k / strict_hit / MRR + 生成層 RAGAS 5指標、
統合eval set 66問）に引きつけてまとめたもの。**「どうすべきか」の最終判断はTMが行う**。
本docは判断材料。

**層ごとに評価設計を分ける**（§1-2 の原則）。指標は §3（検索層 3A / 生成層 3B）、
実験設計は **検索層＝[retrieval-experiment-plan.md](retrieval-experiment-plan.md)**（dense/sparse/QE の 2×2）、
**生成層＝§8**（度数→コード足場の注入）に分離してある。§4-7 は両層に共通の統計・運用。

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

## 3. 指標の選び方（検索層 3A / 生成層 3B で分ける）

**指標は層ごとに別物**。検索層＝LLM不使用の集合・ランキング指標（常時回せる）、
生成層＝RAGAS中心のLLM指標（節目）＋表層 tripwire。混ぜて「BLEUが上がった＝
システム改善」とやると層の帰属（検索が悪いのか生成が悪いのか）が壊れる（§1-2）。

### 3A. 検索層の指標（LLM不使用・常時）

主 recall@k / 副 strict_hit@k / MRR。実装は `evaluate_retrieval()`。

| 指標 | 測るもの | 位置づけ |
| --- | --- | --- |
| **recall@k** | top-kに入った正解記事の割合 | 主指標。下流がk丸ごと読む以上「袋に入ったか」が本質。連続値で部分点が見える |
| **strict&#95;hit@k** | match_type尊重の二値（and=全記事必須） | 「答えられる状態で引けたか」のYes/No。AND質問の壊れ具合を叫ぶ |
| **MRR** | 最初の正解が出た順位の逆数 | 順序品質。single正解質問で自然 |

**採らない指標と理由**（＝「みんな使うから」で入れない）:
- **precision@k**: kを丸ごと生成層に渡す設計では純度より被覆。正解1記事・k=5で上限0.2の床。
- **nDCG**: 段階的関連度ラベルが前提。本プロジェクトは binary集合しか持たない→入れると嘘。
  §2の multi-source 化をしても採らない（binary上のnDCGはMRR/MAPに潰れる）。
- **MAP / F1**: 解釈が重い / precision側の欠陥を相続。recall@k+MRRで実質カバー、n=60で見返り薄い。

（検索層の**実験設計**＝dense/sparse/QE の 2×2 要因計画は
[retrieval-experiment-plan.md](retrieval-experiment-plan.md) に分離）

### 3B. 生成層の指標（LLM＝RAGAS中心・節目 / 表層＝tripwire）

**主判定: RAGAS 5指標**（役割分担）:
- faithfulness = hallucination検出（**最優先**）、answer_relevancy = 質問適合
- context_precision / recall = 検索の質、answer_correctness = end-to-end
- **answer&#95;correctness は ground truth の粒度に強く依存**。低値（0.37）を「システムが悪い」と
  読む前に評価セット側の粒度を疑うのは正しい筋
- judge は生成層とプロバイダごと分離（self-preference bias回避・レート独立。§ config）

**監視 tripwire（無料・決定論・常時、判定用ではない）**:
- **String Presence**: 事実キーワードの包含チェック。音響出力（tempo/key/chord）や
  度数→コードの事実に有効。**ただし ground&#95;truth 基準で判定**（足場の復唱を測らない、§8）。
- **CHRF**: 文字レベルn-gram→トークナイザ不要で日本語・記号表記（「V-I」「5-1」）に頑健。急変アラーム。

**採らない**: Exact Match（自由文で常時≈0）/ BLEU（MT用・単一参照に不適）/
ROUGE（要約用・使うなら ROUGE-L のみ低優先）。理由: **単一gold文 vs 自由文の表層一致は
正しいパラフレーズを罰する**＝answer_correctness が回避する失敗への逆戻り。

**役割固定**: 主判定=RAGAS(意味) / 監視=String Presence・CHRF(表層)。混ぜて
「BLEUが上がった=改善」とやり出すと §5 で禁じた p-hacking に自分で足を突っ込む。

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
| 8 | LLM 3段階法によるボトムアップ層別（§7） | match_typeが見落とす失敗軸の発見 | 小（LLM呼び出しのみ） |
| 9 | forum除外52件への3段階法適用（§7・優先度高） | 顧客課題発見。プラットフォーム構想の需要仮説と接続 | 小〜中 |

順序の目安: 1→2 は他の何をやるにも前提になる基盤。4 は未解決課題
（度数表記の比較質問）に直結。5 は「ハイブリッド検索で改善したか」を
主張したくなった時点で必須になる。

---

## 7. 探索的・質的な補完（徒然研究室メソッドからの示唆）

§1〜6 は仮説駆動（match_type/notation_variant等、故障モードを先に疑ってから層別する）
で一貫しており、統計的厳密性はこの設計の強み。一方でこの型は**予想済みの弱点しか
見つけられない**。徒然研究室（note: tsurezure_cat、データ文化研究）の手法を輸入し、
ボトムアップにカテゴリを浮かび上がらせる工程を補完として加える。

### LLM 3段階法（オープンコーディング→スキーマ固定→根拠付き再分類）

1. **自由タグ抽出**: LLMに対象テキスト全件を読ませ、カテゴリを人間が先に決めずに
   自由にタグを出させる
2. **スキーマ固定**: 出てきたタグを15〜25個程度の主要カテゴリに整理し、各カテゴリの
   定義を文章で固定する
3. **固定スキーマで全件再分析**: 固定した定義で全件を再分類し、スコア数値化に加えて
   **根拠フレーズ（そのタグを付けた理由となる原文の一節）を必ず抽出させる**
   （LLM judgeの hallucination 対策。人間が抜き打ち検証できる形を残す）

### 適用1: eval set 60問のボトムアップ層別

現状の層（match_type/notation_variant/difficulty）はすべて演繹的。3段階法を
eval set 60問にかけ、「定義系/実践アドバイス系/抽象度/口語度」等のデータ由来の
カテゴリを浮かび上がらせ、既存タグと突き合わせる。match_typeより失敗をよく
説明する軸が見つかる可能性がある。コストはほぼゼロ（LLM呼び出し60〜120回程度）。

### 適用2: forum除外52件への適用（優先度高）

Phase 5 で forum由来129件中52件を「記事では答えられない」として除外した
（実体: サイトへの機能要望・誤字報告・学習相談・自作曲の分析依頼等）。
現状この52件は評価セットからの単純な捨て札だが、**「学習者がフォーラムで
本当に求めているものの何割が知識コンテンツでは答えられないか、それは何か」
という問い自体が顧客課題発見の一次データ**になっている。3段階法で52件を
分類し、WHAT vs WHYギャップの顧客リサーチや音楽プラットフォーム構想の
需要仮説と突き合わせる価値がある。

### 適用3: 発信への接続

`experiments/viz_retrieval.py` の埋め込み空間可視化（forum_76963 ケーススタディ等）や
「1クエリ1ベクトルは2つのトピックの近傍に同時にいられない」といった診断は、
そのままnote記事・X投稿の素材になる。実験の採否判断（本ドキュメントの本体）とは
分離し、発信は別途 note 化を検討する（詳細: Second Brain Wiki
「徒然研究室メソッド(違和感→データで課題発見)」）。

### 位置づけの注意

これらは§1〜6の統計的な採否判断プロセスを置き換えるものではない。あくまで
「次に何を疑うべきか」の仮説生成（探索的分析）であり、効果の主張はこれまで通り
paired difference + CI で行う。

---

## 8. 生成層の実験設計: 度数→コード足場の注入

検索層の弱点診断（[retrieval-experiment-plan.md](retrieval-experiment-plan.md) §2）で、
度数表記の比較質問（「5-1と4-1の違い」型）は**検索では構造的に解けない**と結論づけた。
うち **度数→コードの対応は決定論的に計算できる事実**なので、検索に頼らず生成プロンプトへ
直接注入して迂回する。これは hybrid検索 / QE（検索層の打ち手）とは**別レイヤーの第3の打ち手**。

### 研究フレーム（課題 / 仮説 / 根拠 / 結果）
- **課題**: 度数→コードの対応をコーパス検索で取れず、回答のコード名が事実誤りになる。
- **仮説**: 全調のダイアトニック表を生成プロンプトに静的注入すると、度数質問の
  answer_correctness / faithfulness が改善する。
- **根拠**: 度数→コードは一意に計算可能。検索の取りこぼしと無関係に正しい事実を与えられる。
- **結果**: TODO（未計測）

### 設計判断: 「全調バーン注入」を採用（条件付き注入は却下）
- 12メジャー+12マイナー × 7度の**静的表を最初に全部注入**する。
- 却下した案＝「キー確定時だけ注入」: eval set 66問中キー明示は約15問しかなく、
  条件付きだと足場の発火が n≈15 で**測定力不足**。全調注入なら**全問で発火→full set で
  paired diff 可能**、かつキー検出への依存・誤検出で嘘足場を注入するリスクが消える。
- **保守的テスト**: 無関係な23調のノイズ込みで効果が出れば、本番の「検出キー1個」版は
  それ以上に効く（下限測定）。
- **test broad, ship narrow**: 実験＝全調注入（足場効果を検出精度から分離）／
  本番＝audio検出 or 質問明示のキー1個に絞ってノイズを消す。

### 足場の正しさ（最大のリスク）
- **間違った足場＝確信を持って嘘を注入する装置**。検索の「情報なし」より悪い。
- メジャーキーのダイアトニック性質は固定パターン `[maj,min,min,maj,maj,min,dim]`。
  手書きは取り違える（例: キーD の ii は **Em** であって E ではない）。
- → 足場は**決定論的に生成**し、`diatonic("C") == {I:C, ii:Dm, …, vii°:Bdim}` を assert する
  self-check を必ず置く。
- **表記は質問/コーパスに合わせる**（SoundQuestは「5-1」の数字表記→足場も数字表記で出すと
  表記ゆれを直接橋渡しできる）。
- **最小から**: まず7度トライアドのみ。7th / セカンダリードミナントは効果を測ってから足す（YAGNI）。

### 測定
- **主指標**: RAGAS answer_correctness / faithfulness の **paired diff**（足場あり/なし・同一問）。
- **監視**: String Presence（回答が正しいコード名を含むか）。**ただし ground&#95;truth 基準で判定**。
  注入した足場に対して測ると circular（teaching to the test：LLMが足場を復唱しただけを拾う）。
- **層別必須**: 効果はキー/度数を持つ質問に集中する。full-66 平均は概念質問が薄めて
  「有意差なし」に見える → `notation_variant` / キー明示サブセット（約15問、うち
  `generated_001`〜`005` は本仮説に目的特化）で見る。
- **p-hacking guard**: 条件は「足場あり/なし」の2つだけ事前宣言。プロンプト変種を
  10個試して eval set で最良を選ぶ＝過学習（§5 と同じ規律）。

### 切り分けの注意
- 差が出ない場合、「足場不要」ではなく「**LLMが既に知っていた**」可能性がある
  （C/G/D 等の一般的な調）。per-question で**どの調で効いたか**を見る。足場が本当に
  効くのは珍しい調（F#, Db）とマイナーキーの正しい性質（ここを間違えやすい）。

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
