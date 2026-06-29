"""Gemini API を呼び出して音楽理論解説を生成するモジュール。"""
from google import genai
from google.genai import types

import config

SYSTEM_PROMPT = """あなたは音楽理論と楽曲分析の専門家です。
以下の参考資料（音楽理論教材からの抜粋）を根拠に、ユーザーの質問へ日本語で丁寧かつ具体的に答えてください。

ルール:
- 参考資料に書かれている内容を優先して使うこと。
- 参考資料に無い内容を補う場合は「(資料外の一般知識)」と明示すること。
- 楽曲の音響特徴が与えられている場合は、それを資料の理論と結びつけて解説すること。
- コード進行やキーに言及するときは、機能（トニック/サブドミナント/ドミナント等）にも触れること。

[参考資料]
{context}
"""


def explain(query: str, chunks: list[dict], audio_desc: str | None = None) -> str:
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    context = "\n\n---\n\n".join(
        f"[出典: {c['meta'].get('source', '?')}]\n{c['text']}" for c in chunks
    ) or "(参考資料が見つかりませんでした)"

    parts = []
    if audio_desc:
        parts.append(f"# 解析した楽曲の音響特徴\n{audio_desc}")
    parts.append(f"# 質問\n{query}")

    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents="\n\n".join(parts),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(context=context),
        ),
    )
    return resp.text
