"""OpenAI互換API（NVIDIA Build / Gemini）を呼び出して音楽理論解説を生成するモジュール。

生成層プロバイダは config.LLM_PROVIDER（gemini / nvidia、env で切替）で選ぶ。
どちらも OpenAI SDK で叩けるので client の base_url/api_key/model だけ差し替える。"""
from functools import lru_cache

from openai import OpenAI
from openai import InternalServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from music_rag import config


def _make_client() -> tuple[OpenAI, str]:
    """config.LLM_PROVIDER に応じて OpenAI互換クライアントと model 名を返す。"""
    if config.LLM_PROVIDER == "nvidia":
        client = OpenAI(
            api_key=config.NVIDIA_API_KEY,
            base_url=config.NVIDIA_BASE_URL,
            timeout=config.LLM_TIMEOUT_SEC,
        )
        return client, config.NVIDIA_LLM_MODEL
    if config.LLM_PROVIDER == "openrouter":
        client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            timeout=config.LLM_TIMEOUT_SEC,
        )
        return client, config.OPENROUTER_MODEL
    # 既定: gemini（OpenAI互換エンドポイント）
    client = OpenAI(
        api_key=config.GEMINI_API_KEY,
        base_url=config.GEMINI_BASE_URL,
        timeout=config.LLM_TIMEOUT_SEC,
    )
    return client, config.GEN_GEMINI_MODEL

# 検索前の query expansion。実験（条件C）で測ったプロンプトをそのまま使う
# ── 文面を変えると測った recall は担保されない（docs/retrieval-experiment-results.md）。
EXPAND_SYSTEM = """あなたは日本語の音楽理論に詳しい検索クエリ拡張アシスタントです。
ユーザーの質問文を書き換えず、そのまま残したうえで、検索に効く同義語・正規化表記を追記してください。

ルール:
- 元の質問文は1行目にそのまま残す（削除・言い換え禁止）。
- 2行目以降に、度数表記の正規化（例: 「5-1」→「V-I」「ドミナントモーション」「正格終止」）や
  同義の理論用語をスペース区切りで追記する。
- 解説文や長文は書かない。追記は短い語・フレーズのみ。
- 出力はその2ブロックだけ（前置き・後書きなし）。
"""


@lru_cache(maxsize=256)
def expand_query(question: str) -> str:
    """質問文に同義語・正規化表記を追記して検索に渡す文字列を返す。

    QE は検索精度の上積みであって必須経路ではない。失敗・遅延したら黙って
    元の質問文を返す（QE の不調でユーザーの検索そのものを落とさない）。
    評価バッチと違いレート制限の sleep は入れない（対話1件ごとの呼び出しなので）。
    """
    if not config.ENABLE_QE:
        return question
    try:
        client = OpenAI(
            api_key=config.GEMINI_API_KEY,
            base_url=config.GEMINI_BASE_URL,
            timeout=config.QE_TIMEOUT_SEC,
        )
        resp = client.chat.completions.create(
            model=config.QE_GEMINI_MODEL,
            messages=[
                {"role": "system", "content": EXPAND_SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        expanded = (resp.choices[0].message.content or "").strip()
    except Exception:
        return question
    if not expanded:
        return question
    # モデルが元文を落とした場合の保険: 先頭に元質問を戻す
    return expanded if question in expanded else f"{question}\n{expanded}"


SYSTEM_PROMPT = """あなたは音楽理論と楽曲分析の専門家です。
以下の参考資料（音楽理論教材からの抜粋）を根拠に、ユーザーの質問へ日本語で丁寧かつ具体的に答えてください。

ルール:
- 参考資料に書かれている内容を優先して使うこと。
- 参考資料に無い内容を補う場合は「(資料外の一般知識)」と明示すること。
- 楽曲の音響特徴が与えられている場合は、それを資料の理論と結びつけて解説すること。
- コード進行やキーに言及するときは、機能（トニック/サブドミナント/ドミナント等）にも触れること。
- 参考資料が英語の場合も、内容は日本語で説明すること。専門用語や章タイトルに言及するときは英語表記を併記してよい。

[参考資料]
{context}
"""


@retry(
    retry=retry_if_exception_type(InternalServerError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=10, max=60),
    reraise=True,
)
def _generate_content(client: OpenAI, model: str, context: str, contents: str) -> str:
    """生成呼び出し本体。5xx等の一時的なサーバーエラーは指数バックオフでリトライする。"""
    # max_tokens は指定しない。gemini-3.5-flash は思考(thinking)モデルで、
    # max_tokens を小さく絞ると思考トークンが予算を食い潰して可視出力が途中で切れる
    # （1500指定時: 可視66tok / 思考~1430tok で finish_reason=length）。
    # 未指定にしてモデル既定の大きい出力上限に委ね、思考＋本文が両方収まるようにする。
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": contents},
        ],
    )
    return resp.choices[0].message.content


def explain(query: str, chunks: list[dict], audio_desc: str | None = None) -> str:
    client, model = _make_client()

    context = "\n\n---\n\n".join(
        f"[出典: {c['meta'].get('source', '?')}]\n{c['text']}" for c in chunks
    ) or "(参考資料が見つかりませんでした)"

    parts = []
    if audio_desc:
        parts.append(f"# 解析した楽曲の音響特徴\n{audio_desc}")
    parts.append(f"# 質問\n{query}")

    return _generate_content(client, model, context, "\n\n".join(parts))
