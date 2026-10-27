# rag-mcp

本地 RAG 知识库，通过 MCP 提供搜索工具。目前支持 UTF-8 编码的 `.md` 和 `.txt` 文件。

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

## 片段向量化

```python
from dataclasses import asdict

from rag_mcp.chunk import Chunker
from rag_mcp.embed import Embeder

chunks = Chunker().chunk(["./knowledge-base/guide.md", "./knowledge-base/notes.txt"])
embeder = Embeder()

results = embeder.embed(chunks)       # list[Chunk] -> list[Embed]
if chunks:
    result = embeder.embed(chunks[0]) # Chunk -> Embed
    print(result.vector)
    print(result.metadata.text)
    print(asdict(result))
```

单个 `Chunk` 返回一个 `Embed`；列表输入始终返回列表，包括只有一个片段的列表。`embed([])` 返回 `[]`，不请求模型。列表先校验输入类型和正文是否非空，再按顺序逐个请求模型；类型、空白正文和模型错误直接抛出。

向量结果顶层只有 `vector` 和 `metadata`。`EmbedMetadata` 继承 `ChunkMetadata` 的全部字段，并增加 `text`，例如：

```python
{
    "vector": [0.25, -0.5],
    "metadata": {
        "source": "/absolute/path/guide.md",
        "heading_path": ["指南", "安装"],
        "start_line": 2,
        "end_line": 3,
        "start_index": 11,
        "end_index": 15,
        "text": "正文。\n",
    },
}
```

`metadata.text` 保存未经拼接的原始正文，来源和定位字段与输入 chunk 一致。请求向量时，有标题的片段使用 `标题 > 子标题\n\n正文` 提供上下文，无标题片段直接使用正文。结果的标题路径列表独立于输入，修改结果不会改变原 chunk。

```text
src/rag_mcp/embed/
├── __init__.py # 公开 Embeder、Embed、EmbedMetadata
├── schema.py   # vector + metadata，原文 text 放入 metadata
├── embeder.py  # 单个/列表 Chunk 的向量化入口
└── ollama.py   # 字符串向量化请求与响应校验
```

Ollama 默认使用 `http://localhost:11434/api/embed` 和 `qwen3-embedding:0.6b`，配置位于 `embed/ollama.py`。原 `embeding(text)` 字符串入口保留为兼容导出，查询向量化与 `Embeder` 共用同一份请求实现。

## 知识库与 MCP

`embeding_files()` 使用 `Chunker().chunk(files)` 和 `Embeder().embed(chunks)` 处理当前工作目录下 `knowledge-base/` 中的 MD/TXT。片段 ID 采用绝对路径与字符范围，避免同一行内的片段互相覆盖。该函数继续将向量结果转换为检索层已有的 `source`、`heading_path`、`content`、`start_line`、`end_line` 和 `embedding` 字段。

```bash
uv run python -m rag_mcp.server
```

该命令运行 stdio MCP 服务，暴露 `search_knowledge(text)`，返回最多五个相关片段。搜索时请求源码中配置的 Ollama embedding 服务；每次搜索都会重新生成知识库向量。单独调用 `Chunker` 不依赖 Ollama，测试中的模型请求使用 mock。
