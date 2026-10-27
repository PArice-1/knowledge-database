"""将问题转成向量，按余弦相似度返回最相关的片段及得分。"""

import math

from .embeding import embeding
from .embeding_files import EmbeddedChunk


class QueryResult(EmbeddedChunk):
    """保留 EmbeddedChunk 的所有字段，额外包含本次查询的余弦相似度。"""

    score: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算余弦相似度；拒绝空向量、维度不一致、非有限值和零向量。"""
    if not a or len(a) != len(b):
        raise ValueError("向量不能为空，且维度必须相同")
    if any(not math.isfinite(value) for value in a + b):
        raise ValueError("向量必须只包含有限数值")

    length_a = math.hypot(*a)
    length_b = math.hypot(*b)
    if length_a == 0 or length_b == 0:
        raise ValueError("不能计算零向量的余弦相似度")

    # 先归一化再计算点积，避免直接相乘时产生很大的中间值。
    return sum((x / length_a) * (y / length_b) for x, y in zip(a, b))


def query(text: str, data: dict[str, EmbeddedChunk]) -> list[QueryResult]:
    """返回相似度从高到低的前 5 个片段及得分，不足 5 个时返回全部。

    data 使用 embeding_files() 的结果，必须与查询使用相同模型和向量维度。
    每个结果保留片段字段并增加 score，不修改输入；同分时保持原顺序。
    空知识库直接返回 []，不请求模型；输入、向量或模型请求错误直接抛出。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    if not text.strip():
        raise ValueError("text 不能为空或只有空白")
    if not data:
        return []

    # Qwen3-Embedding 推荐为查询添加任务说明；资料正文不添加。
    query_text = (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\n"
        f"Query: {text}"
    )
    query_vector = embeding(query_text)
    # 创建新字典保存本次得分，避免把 score 写入知识库的原始片段。
    results: list[QueryResult] = [
        {
            **chunk,
            "score": cosine_similarity(query_vector, chunk["embedding"]),
        }
        for chunk in data.values()
    ]
    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:5]


if __name__ == "__main__":
    from .embeding_files import embeding_files

    emb_dict = embeding_files()
    emb_list = query("python 如何创建虚拟环境", emb_dict)

    for index, item in enumerate(emb_list, start=1):
        print(f"第 {index} 条 | 相似度：{item['score']:.4f}")
        print("标题：", " > ".join(item["heading_path"]))
        print("来源：", item["source"])
        print("正文：", item["content"])
        print()
