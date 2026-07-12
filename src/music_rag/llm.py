"""NVIDIA Build（OpenAI互換API）を呼び出して音楽理論解説を生成するモジュール。"""
from openai import OpenAI
from openai import InternalServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from music_rag import config

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
def _generate_content(client: OpenAI, context: str, contents: str) -> str:
    """NVIDIA Build呼び出し本体。5xx等の一時的なサーバーエラーは指数バックオフでリトライする。"""
    resp = client.chat.completions.create(
        model=config.NVIDIA_LLM_MODEL,
        max_tokens=config.MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": contents},
        ],
    )
    return resp.choices[0].message.content


def explain(query: str, chunks: list[dict], audio_desc: str | None = None) -> str:
    client = OpenAI(api_key=config.NVIDIA_API_KEY, base_url=config.NVIDIA_BASE_URL)

    context = "\n\n---\n\n".join(
        f"[出典: {c['meta'].get('source', '?')}]\n{c['text']}" for c in chunks
    ) or "(参考資料が見つかりませんでした)"

    parts = []
    if audio_desc:
        parts.append(f"# 解析した楽曲の音響特徴\n{audio_desc}")
    parts.append(f"# 質問\n{query}")

    return _generate_content(client, context, "\n\n".join(parts))
