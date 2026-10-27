"""对外提供知识库查询，返回 JSON 字符串。"""

import json

from .embeding_files import embeding_files
from .query import query


def tool_query(text: str) -> str:
    """查询知识库并返回 JSON 字符串，保留中文，排除 embedding 向量字段。

    当前每次调用都会重新生成知识库向量；查询或模型错误直接抛出。
    """
    knowledges = embeding_files()
    results = query(text, knowledges)
    # 创建对外结果，不删除或修改内部查询结果中的向量。
    public_results = [
        {key: value for key, value in item.items() if key != "embedding"}
        for item in results
    ]
    return json.dumps(public_results, ensure_ascii=False, indent=2, allow_nan=False)
