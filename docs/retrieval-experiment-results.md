# 検索層 2×2 要因実験 — 実行ログと結果（条件 A/B/C）

実行日: 2026-07-21
対象計画: [retrieval-experiment-plan.md](retrieval-experiment-plan.md) §3–§9 / ランナー設計: [../experiments/condition_abc_plan.md](../experiments/condition_abc_plan.md)
eval set: `data/eval/eval_set_merged.json`（66問, frozen） / k=5 / Base = structure collection, dense-only（`data/eval/scores_20260719.json` の `structure`）

このドキュメントは「実際に条件 A/B/C を動かして CSV に落とす」作業の**過程・判断・つまずき・結論**を残す。数値の再現物は CSV / JSON（末尾のファイル一覧）にあり、ここはそれらを**なぜその形で得たか**の記録。

---

## 0. 結論の要約（先に）

- **A/B/C いずれも、全指標で 95%CI が 0 をまたぐ → n=66 では Base に対する有意差を主張できない。**
  fixed vs structure（過去 n=60）と同じ結論構造。
- 唯一のニアミス信号は **条件C（dense+sparse+QE）の recall@5**: 平均差 **+0.076**, 95%CI **[−0.010, +0.162]**, p=0.099, 改善11/悪化4。**n がもう少しあれば有意になりそうな唯一の勾配**。
- **相互作用は弱い相乗**（干渉なし）: (C−Base) − [(A−Base)+(B−Base)] = +0.015（recall）。§8 マトリクスの「A↑ B↑ C≈A+B（相補的）」寄り。
- sparse 単独（A）は **MRR がむしろ悪化寄り**（8改善/12悪化）＝順位に無関係な語彙一致を上位に混ぜて薄める副作用の兆候。QE と組む C ではそれが解消。
- **計画が主指標に宣言した「表記ゆれ層（notation_variant）の recall」は未評価**（タグ未付与）。ここは全66問平均での近似にとどまる。→ 次の最有力手。

---

## 1. 実行した条件

| 条件 | dense | sparse | QE | collection |
| --- | --- | --- | --- | --- |
| Base | ✓ | | | `music_theory_structure`（既存, 2,355点） |
| A | ✓ | ✓ | | `music_theory_hybrid`（本実験で新規構築） |
| B | ✓ | | ✓ | `music_theory_structure` |
| C | ✓ | ✓ | ✓ | `music_theory_hybrid` |

QE モデル = `gemini-3.1-flash-lite`（expansion: 元文保持 + 度数正規化・同義語追記）。融合 = Qdrant Query API の RRF（Prefetch limit=50）。

---

## 2. 実行環境と、そこから来た制約（過程の記録）

作業機は **8GB RAM / ディスク空き ~6GB** の 1 台。プロジェクトの既知ハザード（memory）がそのまま効いた：

- **BGE-M3 embed と Docker/Qdrant の同時実行 → スワップ膨張 → Docker VM クラッシュの前例**がある。
- **BGE-M3 を MPS に載せると 8GB では OOM abort。CPU 固定が必須。**

このため hybrid collection の構築を **3 フェーズに時間分離**した（`experiments/ingest_hybrid.py`）：

1. **export（Docker UP）**: `music_theory_structure` を scroll し、各点の `id / payload / dense ベクトル` を JSONL に dump（2,355点, 34MB）。Qdrant は軽いので embed と同時に走らせない。
2. **sparse（Docker DOWN）**: Qdrant を止めてメモリを空け、BGE-M3 を **CPU 固定**で回して各チャンク text の sparse（lexical weights）だけ計算。checkpoint 付き（途中失敗しても再開可）。実測 2,355点で完走。
3. **upsert（Docker UP）**: named vectors（dense + sparse）で `music_theory_hybrid` を作成し upsert。

### 設計上の核心的判断: dense は「再埋め込み」せず「コピー」

計画 §7.1 は「hybrid の dense は Base と bit 一致していること（でないと sparse の効果が分離できない）」を要求する。素直にやると 2,355 チャンクを再 embed して一致検証する必要があるが、**既存 structure collection に dense ベクトルが既にある**。そこで：

- structure から dense を**そのままコピー**して hybrid に入れ、BGE-M3 では **sparse のみ**新規計算した。
- 結果、dense bit 一致は**自明に保証**（検証でもランダム5点 5/5 一致）。かつ dense の再計算コストがゼロになった。

本番 `embedder.py` / `retriever.py` / `query_pipeline.py` は一切変更していない（実験隔離の原則）。実験専用の retrieve 経路は `experiments/retrieval_exp_common.py` にのみ存在する。

---

## 3. つまずきと対処（QE = Gemini 呼び出し周り）

条件 B/C は検索前に Gemini を 1 回/問 叩く。無料枠で以下が順に起きた：

1. **429 RateLimitError**: `gemini-3.1-flash-lite` の無料枠は **15 RPM**。66問を連射して即死。
   → 呼び出し間隔を **12 RPM 相当（5s）にスロットル** + 429 は `retryDelay` を尊重してリトライ。
2. **APITimeoutError（1問で 9 分無駄）**: 既定 client timeout が 90s。無料枠がまれに stall すると、リトライ6回 ×（90s hang + 待機）で 1 問に ~13 分かけて死ぬ。
   → QE 用 client の timeout を **25s に短縮**（QE は本来 ~1.5s で返る）。「hang したら早く諦めて retry」に切替。
3. **特定の 1 問が stall して run 全体を道連れ**（forum_67471）。単独で試すと 4.7s で正常応答 → 恒常的な毒ではなく無料枠の一時的な stall 窓。
   → QE がリトライ上限まで失敗したら **その問だけ raw question にフォールバック**して継続（失敗は cache しない → 後の再実行で再試行）。1 問の stall で 65 問を落とさない。

### 効いた設計: QE 結果のディスク cache

`experiments/_hybrid_scratch/qe_cache.json` に「質問文 → 展開後クエリ」を保存。**QE は質問ごとに一生に一度だけ**呼ぶ：

- **B → C で使い回し**: C は同じ66問なので Gemini 呼び出しゼロ（cache 読むだけ）。
- **失敗 run をまたいで再開**: B が 48/66 で落ちても、再実行は残り18問だけ叩く。RPD（memory: 失敗リクエストも RPD を消費する）を無駄にしない。

（ユーザーの「一度出した QE をB/Cで使い回せば？」という指摘は、この cache 設計そのもの。条件間だけでなく失敗再実行もまたいで効く形になっている。）

---

## 4. 結果 — 集計（vs Base 0.599）

| | recall@5 | strict_hit | MRR | AND recall (n=27) | OR (n=19) | single (n=20) |
| --- | --- | --- | --- | --- | --- | --- |
| **Base** dense | 0.599 | 0.500 | 0.562 | 0.352 | 0.684 | 0.850 |
| **A** dense+sparse | 0.621 | 0.530 | 0.555 | 0.407 | 0.684 | 0.850 |
| **B** dense+QE | 0.636 | 0.500 | 0.614 | 0.407 | 0.684 | 0.900 |
| **C** 両方 | **0.674** | 0.545 | 0.615 | 0.426 | 0.684 | **1.000** |

- C は全指標で最良。single 層が 0.85 → **1.00**。AND 層も 0.352 → 0.426 と少し動いた（依然低いが）。
- OR 層はどの条件でも 0.684 で不変。

---

## 5. 結果 — paired difference（本命の判断材料）

同一66問での per-question 差分 + bootstrap 95%CI（10,000リサンプル）+ Wilcoxon 符号順位検定。全体平均比較より感度が高い。

| 比較 | 指標 | 平均差 (b−a) | 95%CI | 改善/悪化/不変 | p |
| --- | --- | --- | --- | --- | --- |
| Base→**A** | recall | +0.023 | [−0.023, +0.076] | 4/1/61 | 0.56 |
| | strict_hit | +0.030 | [−0.030, +0.091] | 3/1/62 | 0.63 |
| | mrr | **−0.007** | [−0.063, +0.055] | 8/12/46 | 0.65 |
| Base→**B** | recall | +0.038 | [−0.035, +0.111] | 8/4/54 | 0.38 |
| | strict_hit | **0.000** | [−0.076, +0.076] | 3/3/60 | 1.00 |
| | mrr | +0.052 | [−0.034, +0.139] | 13/12/41 | 0.32 |
| Base→**C** | **recall** | **+0.076** | **[−0.010, +0.162]** | 11/4/51 | **0.099** |
| | strict_hit | +0.046 | [−0.046, +0.136] | 6/3/57 | 0.51 |
| | mrr | +0.053 | [−0.035, +0.147] | 16/12/38 | 0.33 |

### 読み取り

- **全部 CI が 0 をまたぐ**。計画 §6-4 の基準（CI が 0 をまたがない時のみ「改善」と呼ぶ）に照らし、有意な改善は 1 つも主張できない。
- **不変が recall/strict_hit で 5〜6割**＝大多数の質問で検索結果が動いていない。効果は「動いた少数の質問」に集中し、そこでは C が改善優勢（11/4）。
- **C の recall だけがニアミス**（下端 −0.010, p=0.099）。改善が悪化の 2.75 倍。**追加の n か、層を絞れば有意化しうる唯一の信号**。
- **A の MRR は悪化寄り**（8/12）: sparse は recall をわずかに上げるが順位はむしろ乱す。単独採用の分が悪い根拠。
- **相互作用（recall）= +0.015 の弱い相乗**。QE（意味側）と sparse（語彙側）は干渉せず、わずかに足し合わさる。

---

## 6. 限界（この結果で言えないこと）

- **主指標未評価**: 計画は A/B の採否を「notation_variant（度数・記号表記ゆれ）タグ付き質問の recall」で判断すると事前宣言した。タグが未付与のため、ここは全66問平均での近似にとどまる。**C のニアミスが表記ゆれ質問で起きているのか**は、タグを付けるまで確定できない。→ 次の最有力手。
- **AND 層は想定通り動かず**（計画 §2 診断1）。1クエリ=1ベクトルの構造的天井で、sparse/QE では原理的に直らない。失敗ではなく想定内。分解検索（Phase 2）の担当。
- **QE のフォールバック**: run 途中で raw question に落ちた問が発生しうる（毎回同一ではない＝Gemini の stall タイミング依存）。最終的には全66問が cache 済みで、CSV の `expanded_query` 列で各問の実際の展開文を確認できる。
- **RAGAS（生成層 context_precision）は未実施**。structure の真価は過去 RAGAS でのみ観測されている（cp +0.134）。retrieval 指標は chunking 戦略も本実験の条件も区別しにくい。最良条件（C）vs Base の RAGAS は節目で 1 ペアだけ回す価値がある（計画 §5, §9-6）。

---

## 7. 次の一手（優先順）

1. **notation_variant タグ付与 → freeze**（計画 §5 前提タスク）。C recall のニアミスを「表記ゆれ層」で撃ち直す。層の n=15〜20 でも大きな効果なら検出可能な設計。
2. C（最良）vs Base の **RAGAS context_precision の paired diff**（生成=flash-lite / judge 分離）。
3. AND 層は Phase 2（クエリ分解→並列検索→マージ）へ。本実験では触らない。

---

## 8. 生成物ファイル一覧

| ファイル | 内容 | Git |
| --- | --- | --- |
| `data/eval/scores_condition_a_20260721.{json,csv}` | 条件A per-question + 層別集計 | gitignore（data/eval） |
| `data/eval/scores_condition_b_20260721.{json,csv}` | 条件B（`expanded_query` 列あり） | 同上 |
| `data/eval/scores_condition_c_20260721.{json,csv}` | 条件C（`expanded_query` 列あり） | 同上 |
| `data/eval/scores_conditions_combined_20260721.json` | base+A+B+C を1ファイルに（paired_diff 入力・再現用） | 同上 |
| `experiments/ingest_hybrid.py` | hybrid collection 構築（export/sparse/upsert 3フェーズ） | コミット対象 |
| `experiments/_hybrid_scratch/` | 中間物（dense dump / sparse / qe_cache.json）。再実行用に残置 | gitignore 推奨 |
| Qdrant `music_theory_hybrid` | 2,355点, named dense(copy)+sparse。dense bit一致 5/5 検証済み | ローカルのみ |

CSV は 12 列: `id, question, match_type, source, difficulty, notation_variant, recall, strict_hit, reciprocal_rank, expected, retrieved, expanded_query`（list 列は `|` 区切り）。

---

## 9. コード変更点（本番は不変）

- `experiments/retrieval_exp_common.py`: CSV 出力（`_write_csv`）、QE の 12 RPM スロットル + 429/timeout リトライ + ディスク cache + 1問フォールバックを追加。
- `experiments/ingest_hybrid.py`: 新規（hybrid 構築）。
- 本番パイプライン（`src/music_rag/*`）・既存 collection（`music_theory_structure`）は**無変更**。
