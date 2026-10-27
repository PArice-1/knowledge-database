"""按 TXT 独立规则递归切分，不解释 Markdown 语法。"""

from pathlib import Path

from chonkie import RecursiveChunker

from ._shared import SourceDocument
from .rules import DEFAULT_CHUNK_SIZE, MIN_CHARACTERS_PER_CHUNK, TEXT_RULES
from .schema import Chunk


def text_parser(file: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[Chunk]:
    """读取 UTF-8 文本，返回 heading_path 为 [] 的原文片段。"""
    document = SourceDocument.read(file)
    chunker = RecursiveChunker(
        tokenizer="character",
        chunk_size=chunk_size,
        rules=TEXT_RULES,
        min_characters_per_chunk=MIN_CHARACTERS_PER_CHUNK,
    )
    return document.split(chunker, 0, len(document.text), [])
