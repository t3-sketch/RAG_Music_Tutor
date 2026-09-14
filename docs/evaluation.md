# 評価記録: fixed vs structure chunking（2026-07）

<a id="evidence"></a>
## 判断から実物を確認する

2026-09-14に追加した事後の案内です。以下の2026-07の報告値を再計算したものではありません。

- **課題・結果**：hit-rate@5は両条件0.85でも、context precisionの平均は0.653 → 0.788。下の平均表と質問別表を参照。
- **判断**：正解記事の取得と、取得文脈の質を分けて見る。構造分割を採用したが、20問の平均差だけで統計的な優位性は結論しない。
- **評価コード**：[検索の質問別採点](https://github.com/t3-sketch/RAG_Music_Tutor/blob/64bc594157cec6cea2d0caeae655efb4ae737d6e/experiments/evaluation.py#L122)／[文脈・回答を別々に採点](https://github.com/t3-sketch/RAG_Music_Tutor/blob/64bc594157cec6cea2d0caeae655efb4ae737d6e/experiments/evaluation.py#L289)。現在の公開版の仕組みを示し、2026-07実行版の保証とはしない。
- **設計資料**：[experiment-design.md](experiment-design.md)の§3B・§4。これは後日の設計ガイドであり、この比較の事前登録ではない。古い指標選定方針も含む。
- **非公開・未検証**：分割・取り込みコード、コーパス、評価データと生出力。当時の数値の独立再計算、judgeの反復測定、平均差の有意性検証はこの案内の対象外。

[READMEへ戻る](../README.md#評価から判断したこと)


- fixed: `music_theory`（固定長800字chunking、RAGAS 2026-07-01実施）
- structure: `music_theory_structure`（構造ベースchunking、RAGAS 2026-07-02実施）
- n=20問（silver Q&Aセット）、judge: gemini-3.1-flash-lite、生成層: gemini-3.5-flash
  （self-preference bias回避のため judge と生成層は別モデル）
  ※ この記録は当時の構成での実測。その後、生成層は NVIDIA Build（`meta/llama-3.3-70b-instruct`）へ
  移行済み（Gemini無料枠のRPD枯渇回避）。judge は bias 回避のため引き続き Gemini 側。
  下表の数値は上記日付時点の chunking A/B 比較記録として保持している。

検索層の評価は「hit-rate@k / MRR（LLM不使用・常用）」と「RAGAS（LLM judge・節目のみ）」の2層構成。

## hit-rate / MRR（retrieval層のみ）

| strategy | hit_rate@5 | MRR |
|---|---|---|
| fixed | 0.85 | 0.792 |
| structure | 0.85 | 0.800 |

hit-rateでは差が見えない。以下のRAGASで初めてchunking品質の差が可視化された。

## サマリ（平均）

| metric | fixed | structure | diff |
|---|---|---|---|
| faithfulness | 0.804 | 0.764 | -0.040 |
| answer_relevancy | 0.881 | 0.878 | -0.003 |
| answer_correctness | 0.361 | 0.369 | +0.008 |
| context_precision | 0.653 | 0.788 | +0.134 |
| context_recall | 0.667 | 0.717 | +0.050 |

## per-question: context_precision / context_recall

★ = fixedで cp または cr が 0.0 だった質問（固定長chunkingの弱点）

| question | cp fixed→structure | cr fixed→structure | |
|---|---|---|---|
| コードの機能ってなに | 0.70 → 1.00 | 1.00 → 1.00 |  |
| カデンツとは | 0.89 → 1.00 | 1.00 → 1.00 |  |
| 3mってトニックかサブドミナントどっち | 0.00 → 1.00 | 0.00 → 1.00 | ★ |
| 5-1と4-1の違い | 0.00 → 0.00 | 0.00 → 0.00 | ★ |
| セカンダリードミナントってなに | 1.00 → 1.00 | 0.50 → 0.50 |  |
| 機能和声論において、なぜIVよりもVの方がより強い緊張感を生み出し、展開のピークとなるのですか？ | 1.00 → 0.50 | 1.00 → 1.00 |  |
| 「Vm7→I7→IV」の進行について | 0.33 → 0.50 | 0.50 → 0.50 |  |
| 傾性音とは | 0.87 → 1.00 | 1.00 → 1.00 |  |
| モチーフって何 | 0.33 → 0.70 | 0.50 → 0.50 |  |
| 7-1と4-3の解決の違い | 0.00 → 0.00 | 0.00 → 0.00 | ★ |
| メロディのの三段階モデルとリクイデーション | 0.70 → 1.00 | 1.00 → 1.00 |  |
| メロディ展開における「断片化（Fragmentation）」とはどのようなテクニックで、「制服のマネキン」ではどのように活用されていますか？ | 1.00 → 0.83 | 1.00 → 1.00 |  |
| メロディにおいて傾性に逆らうとどうなる？またどんな使い分けがある | 1.00 → 0.50 | 0.67 → 0.33 |  |
| ユーミンの「春よ、来い」はどんな構造になっている？ | 1.00 → 1.00 | 1.00 → 1.00 |  |
| バックビートとは | 1.00 → 1.00 | 1.00 → 1.00 |  |
| シンコペーションとは | 0.25 → 0.83 | 0.50 → 0.50 |  |
| Anticipationとは | 0.00 → 1.00 | 0.67 → 0.33 | ★ |
| スケールって何？ | 1.00 → 1.00 | 0.00 → 1.00 | ★ |
| 「調性（Tonality）」とは | 1.00 → 0.89 | 1.00 → 1.00 |  |
| 調性がない音楽 | 1.00 → 1.00 | 1.00 → 0.67 |  |

## 知見

- hit-rate@5 / MRR では両者に差が出ない（0.85/0.792 vs 0.85/0.80）が、RAGASの context_precision で +0.134 の改善が可視化された。
- 比較形式質問「3mってトニックかサブドミナントどっち」が cp/cr 0.0→1.0 に回復。
- 「5-1と4-1の違い」「7-1と4-3の解決の違い」は依然 0.0。chunkingでは解けない検索課題（度数表記のゆれ・複数記事にまたがる比較）。
- answer_correctness は横ばい（粒度ミスマッチ疑い、生成層/評価セット側の課題）。
