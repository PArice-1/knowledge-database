"""将知识库中的 Markdown/TXT 片段及其向量打包为字典。"""

from pathlib import Path
from typing import TypedDict

from .chunk import Chunker, search_files
from .embed import Embeder


class EmbeddedChunk(TypedDict):
    source: str
    heading_path: list[str]
    content: str
    start_line: int
    end_line: int
    embedding: list[float]


def embeding_files(root: Path = Path("./knowledge-base")) -> dict[str, EmbeddedChunk]:
    """递归处理知识库，返回 chunk 标识到原文、来源及向量的映射。

    默认目录相对于当前工作目录。key 由绝对文件路径和原文字符范围组成，
    避免同名章节或同一行内的片段互相覆盖；行范围从 0 开始、结束位置不包含在内。
    只返回内存中的字典，不写文件；路径、解析和模型请求错误直接抛出。
    """
    files = search_files([root])
    chunks = Chunker().chunk(files)
    embedded = Embeder().embed(chunks)
    result: dict[str, EmbeddedChunk] = {}

    for item in embedded:
        metadata = item.metadata
        chunk_id = f"{metadata.source}:{metadata.start_index}:{metadata.end_index}"
        result[chunk_id] = {
            "source": metadata.source,
            "heading_path": list(metadata.heading_path),
            "content": metadata.text,
            "start_line": metadata.start_line,
            "end_line": metadata.end_line,
            "embedding": item.vector,
        }

    return result


if __name__ == "__main__":
    data = embeding_files()
    print(f"已处理 {len(data)} 个 chunk")
