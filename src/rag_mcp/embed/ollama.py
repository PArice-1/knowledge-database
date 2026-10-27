"""通过本地 Ollama 将字符串转换为向量。"""

import json
import math
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "qwen3-embedding:0.6b"
TIMEOUT_SECONDS = 120


def embed_text(text: str) -> list[float]:
    """返回一段非空文本的向量，供片段向量化和查询向量化共用。

    输入类型错误时抛出 TypeError，空白文本或向量格式错误时抛出 ValueError。
    网络、HTTP 和 JSON 解析错误直接向调用方抛出。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    if not text.strip():
        raise ValueError("text 不能为空或只有空白")

    payload = {
        "model": MODEL,
        "input": text,
        "truncate": False,
    }
    request = Request(
        url=OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        result = json.load(response)

    embeddings = result.get("embeddings") if isinstance(result, dict) else None
    if not isinstance(embeddings, list) or len(embeddings) != 1:
        raise ValueError("Ollama 响应必须包含一个向量")

    vector = embeddings[0]
    if not isinstance(vector, list) or not vector:
        raise ValueError("Ollama 返回的向量必须是非空列表")
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError("Ollama 返回的向量必须只包含有限数值")

    return [float(value) for value in vector]
