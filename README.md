# rag-mcp

本地知识库文件切分，支持 UTF-8 编码的 `.md` 和 `.txt` 文件。

## 安装与验证

```bash
uv sync --frozen
uv run python -m unittest discover -s tests -v
```

## 文件切分

```python
from dataclasses import asdict

from rag_mcp.chunk import Chunker

chunker = Chunker(chunk_size=2048)
chunks = chunker.chunk("./knowledge-base/guide.md", ".md")
notes = chunker.chunk("./knowledge-base/notes.txt", "txt")
# 后缀可省略，从文件名推断；支持大写扩展名。
notes = chunker.chunk("./knowledge-base/notes.txt")
# 文件列表逐个推断后缀，可混合 MD/TXT，按输入顺序合并为 list[Chunk]。
chunks = chunker.chunk(["./knowledge-base/guide.md", "./knowledge-base/notes.txt"])

for chunk in chunks:
    print(chunk.text)
    print(chunk.metadata.heading_path)
    print(asdict(chunk))
```

单文件和文件列表都返回一个合并后的 `list[Chunk]`，空列表返回 `[]`。传入显式 `suffix` 时对列表中的所有文件生效；省略时按各文件扩展名分别选择解析函数。每个片段只有 `text` 和 `metadata` 两个字段。`asdict()` 后的结构：

```python
{
    "text": "正文。\n",
    "metadata": {
        "source": "/absolute/path/guide.md",
        "heading_path": ["指南", "安装"],
        "start_line": 2,
        "end_line": 3,
        "start_index": 11,
        "end_index": 15,
    },
}
```

`source` 是展开 `~`、解析相对路径和符号链接后的绝对路径。`heading_path` 从父标题排列到子标题；前言、无标题文档及 TXT 使用 `[]`。

行号和字符偏移都从 **0** 开始，结束位置不包含在范围内。行范围覆盖片段涉及的所有行；多个片段可能位于同一行，应使用 `原文[start_index:end_index]` 精确还原正文。字符偏移按 Python 字符计算，不是 UTF-8 字节偏移。原文的 CRLF、CR 和 LF 换行会保留。

### 目录与规则

```text
src/rag_mcp/chunk/
├── __init__.py  # 公开入口
├── chunker.py   # Chunker 类，按后缀调度
├── rules.py     # Markdown、TXT 各自的 RecursiveRules
├── schema.py    # Chunk、ChunkMetadata
├── markdown.py # Markdown 标题解析与递归切分
├── text.py     # TXT 递归切分
├── _shared.py  # 原文读取、偏移和行号映射
└── files.py    # 文件发现、排序与去重
```

切分使用 Chonkie `RecursiveChunker`，采用 `character` tokenizer，默认每片最多 **2048 个字符**。这是固定规则切分，不调用模型，也不生成语义向量。规则集中维护在 `rules.py`：

- Markdown：先用 Markdown 语法识别顶层 h1–h6 和 Setext 标题；再在每个章节内依次尝试段落、换行、中英文句末标点、从句标点、空格，最后按字符数切分。
- TXT：不解释 Markdown 语法，依次尝试段落、换行、中英文句末标点、空格，最后按字符数切分。

只有超过大小限制的部分才进入下一层规则。Markdown 标题保存在元数据中，正文不重复附加标题；代码块、列表和引用内部的标题不建立章节。超长代码块也会继续切分。空白章节和空白片段不生成结果，空文件返回 `[]`。不支持的后缀抛出 `ValueError`，文件读取及解码错误直接向上传递。

原 `chunk.py` 已拆成此包。`markdown_parser()`、`parse_md_list()` 和 `search_md_files()` 入口继续保留；原 `ChunkInfo` 的扁平字段改为 `Chunk.text` 与 `Chunk.metadata`，标题路径从 tuple 改为 list。`search_files()` 可同时发现 MD 和 TXT。
