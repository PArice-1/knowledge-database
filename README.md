# rag-mcp

本地持久化 RAG 知识库：将多个目录/文件中的资料统一写入 Qdrant，通过 MCP 提供语义搜索、关键词搜索、文档读取和状态查询。目前支持 UTF-8 编码的 `.md` 和 `.txt` 文件。

## 安装与验证

```bash
uv sync --frozen
uv run python -m unittest discover -s tests -v
```

## 快速开始

先启动本地 Ollama 并安装 `qwen3-embedding:0.6b`，然后构建知识库、启动 MCP：

```bash
# 将多个目录写入同一个知识库，也可混合传入具体 MD/TXT 文件。
uv run rag-build-kb ./knowledge-base ./other-docs --db-path ./.qdrant --collection knowledge

# 使用与构建时相同的存储目录和集合，通过 stdio 提供工具。
uv run rag-mcp --db-path ./.qdrant --collection knowledge
```

`./other-docs` 是示例路径，需替换为实际存在的目录。语义搜索只生成问题向量，不会重新切分或向量化全部资料；其他工具不依赖 Ollama。构建与工具调用不要同时操作同一个本地数据库。

### 项目结构

```text
src/rag_mcp/
├── build_kb.py          # 多目录/文件构建入口和 CLI
├── config.py            # 构建与服务共用的默认存储配置
├── chunk/               # MD/TXT 切分
├── fliter/              # 切分前文件去重
├── embed/               # Ollama 向量化
├── vector_databbase/    # Qdrant 持久化封装
└── mcp/
    ├── __init__.py       # 公开 KnowledgeBase
    ├── knowledge_base.py # 持久化检索、文档读取和统计
    ├── server.py         # MCP 服务工厂及 stdio 入口
    └── tools/
        ├── search.py     # 语义搜索、关键词搜索
        ├── documents.py  # 文档列表、文档和片段读取
        ├── knowledge.py  # 知识库统计
        └── _shared.py    # JSON 输出和只读工具声明
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

单文件和文件列表都返回一个合并后的 `list[Chunk]`，空列表返回 `[]`。列表参数支持 `list[Path]`、`list[str]` 和混合的 `list[str | Path]`，使用泛型声明以兼容静态类型检查。传入显式 `suffix` 时对列表中的所有文件生效；省略时按各文件扩展名分别选择解析函数。每个片段只有 `text` 和 `metadata` 两个字段。`asdict()` 后的结构：

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

Ollama 默认使用 `http://localhost:11434/api/embed` 和 `qwen3-embedding:0.6b`，配置位于 `embed/ollama.py`。字符串向量化统一使用 `rag_mcp.embed.ollama.embed_text(text)`，查询与 `Embeder` 共用同一份请求实现。旧的 `embeding.py` 兼容入口已删除。

## 向量数据库

`vector_databbase/` 提供 Qdrant 本地持久化封装，公开 `VectorDatabase`。初始化传入数据库目录，`self.client` 是底层 `QdrantClient`，不需要单独启动 Qdrant 服务。

```python
from rag_mcp.embed import Embed, EmbedMetadata
from rag_mcp.vector_databbase import VectorDatabase

embeds = [
    Embed(
        vector=[1.0, 0.0, 0.0],  # 示例向量；实际使用 Embeder 的输出
        metadata=EmbedMetadata(
            source="/knowledge-base/guide.md",
            heading_path=["指南", "安装"],
            start_line=2,
            end_line=3,
            start_index=11,
            end_index=15,
            text="正文。\n",
        ),
    ),
]

with VectorDatabase(db_path="./.qdrant") as db:
    # 创建新集合时执行一次；已有集合可直接 upsert。
    db.create_collection(collection_name="knowledge", vector_size=len(embeds[0].vector))
    point_ids = db.upsert(collection_name="knowledge", embeds=embeds)
```

- `create_collection(collection_name, vector_size, embedding_model=...)` 创建空集合，使用 `Cosine` 余弦相似度，返回底层客户端的布尔结果。可选的模型标识保存在集合元数据中，构建和语义搜索会检查是否匹配；同名集合已存在时抛出异常，不清空或覆盖原集合。
- `upsert(collection_name, embeds)` 接收 `list[Embed]`，将 `vector` 保存为 Point 的向量、`asdict(metadata)` 保存为 payload，等待写入完成后按输入顺序返回 Point UUID 列表。可选的 `processing_keys` 参数接收 `source -> 文件处理标识`，同时保存 `processing_key`、`ingestion_id`、文件内 `chunk_index` 和 `chunk_count`；此时必须一次传入每个文件的全部片段。省略时保持原有元数据结构。`upsert(..., [])` 返回 `[]`，不访问数据库；输入类型和数据库错误直接抛出。
- Qdrant 按 Point ID 判断新增或覆盖，不按正文自动去重。本封装每次生成新 UUID，因此直接重复调用 `upsert` 仍会新增记录。下面的 `Fliter` 在切分、向量化和写入之前执行文件去重。
- 数据保存在 `db_path`，关闭后重新打开仍可读取。支持 `with VectorDatabase(db_path) as db`：正常结束或抛出异常都会自动调用 `close()` 释放文件锁，块内异常继续向上传递。也可手动调用 `close()`；同一路径不要同时打开多个本地客户端。

MCP 的检索和读取也使用这个持久化数据库，不再使用每次查询时重新生成向量的内存知识库。

### 构建并验证知识库

`build_kb` 接收 `list[str | Path]`，目录和具体文件可以混合传入：

```python
from pathlib import Path
from rag_mcp.build_kb import build_kb

written = build_kb(
    ["./knowledge-base", Path("./other-docs"), "./notes/selected.md"],
    db_path="./.qdrant",
    collection_name="knowledge",
    chunk_size=2048,
)
print(f"本次新增 {written} 个片段")
```

- 目录递归扫描 MD/TXT，显式传入的不支持类型会报错。
- 重复路径、重叠目录和符号链接别名会合并；最终文件按绝对路径排序。
- 排除 `db_path` 内的文件，所有输入统一写入同一个存储目录和集合。
- `db_path` 默认是当前工作目录下的 `./knowledge-base/storage`，不会随输入目录改变；集合默认 `knowledge`。
- 返回本次新增片段数，不返回内存向量。空列表、空目录、无正文或全部重复时返回 `0`，不调用模型。
- 任意输入路径无效时，在向量化和写入前报错，不悄悄忽略；`paths` 必须是列表，单个路径需写成 `[path]`。

CLI 与 Python 入口使用同一实现：

```bash
uv run rag-build-kb ./knowledge-base ./other-docs ./notes/selected.md \
  --db-path ./.qdrant --collection knowledge --chunk-size 2048
# 等价模块入口：
uv run python -m rag_mcp.build_kb ./knowledge-base ./other-docs --db-path ./.qdrant
```

构建依次执行 `Fliter → Chunker → Embeder → VectorDatabase`。向量维度取自实际模型输出；写入后关闭并重新打开数据库，校验新增记录数，并核对本次首条片段正文。不会打印资料正文。

重复文件在切分前过滤，但仍需读取字节计算哈希。修改文件或切分配置后追加新版本，不自动删除旧记录；更换 embedding 模型必须使用新集合，不能将不同模型混入已记录模型的集合，即使维度相同也不允许。历史集合没有模型元数据时，无法自动验证模型，需自行保证兼容或重建到新集合。

离线联调测试使用真实切分和临时 Qdrant 数据库，mock Ollama 请求，并在关键场景监测 `Chunker` 调用，覆盖多目录与具体文件混合输入、重叠路径、排除存储目录、重复构建不切分、改名、文件修改、配置变更、模型隔离、旧记录重建、模型失败及部分写入后重试：

```bash
uv run python -m unittest discover -s tests -p test_build_kb.py -v
```

### 过滤规则与持久化去重

```text
src/rag_mcp/fliter/
├── __init__.py  # 公开 Fliter、FileSelection、ProcessingProfile 和过滤函数
├── schema.py    # 文件选择结果和处理配置
├── fliter.py    # 注册规则、按顺序执行、分页读取 Qdrant 完整性元数据
└── duplicate.py # 文件处理标识计算和重复文件过滤
```

`Fliter(db, collection_name).filter(files)` 接收文件路径序列，返回 `FileSelection(files, processing_keys)`：`files` 是待切分的绝对路径列表，`processing_keys` 是绝对路径到处理标识的映射。不传数据库时只在当前批次去重；相同文件按输入顺序保留第一份，副本路径不会另外存为来源。

```python
from rag_mcp.chunk import Chunker
from rag_mcp.embed import Embeder, ollama
from rag_mcp.fliter import Fliter, ProcessingProfile

chunker = Chunker(chunk_size=2048)
profile = ProcessingProfile(chunk_size=chunker.chunk_size, embedding_model=ollama.MODEL)
selection = Fliter(db, "knowledge", profile=profile).filter(files)
chunks = chunker.chunk(selection.files)
embeds = Embeder().embed(chunks)
db.upsert("knowledge", embeds, processing_keys=selection.processing_keys)
```

处理标识采用 `sha256:file-v1:` 前缀，基于**原始文件字节的 SHA-256、解析器类型（MD/TXT）、chunk_size、切分逻辑版本、embedding 模型及输入拼接逻辑版本**计算，不包含路径。文件改名、复制到其他路径后仍能识别；相同字节的 MD 和 TXT 不会混为一份，扩展名大小写不影响判断。原始内容中任何字节变化都可能触发重新处理，不进行语义或切分后正文去重。不同文档共有的片段、文档内部重复段落均保留。

`ProcessingProfile` 必须与实际切分和向量化配置一致；`build_kb` 自动使用当前 `Chunker.chunk_size` 和 `ollama.MODEL`。修改切分规则或模型输入拼接逻辑时，需同步更新 `chunk_version` 或 `embedding_version`；同名模型的权重变更也需要人为更新版本标识。更换模型应使用新集合，不能仅因维度相同就混用向量；现有集合维度也不会自动迁移。

默认规则为 `remove_duplicates`；自定义规则的形式为 `(selection: FileSelection, stored_keys: set[str]) -> FileSelection`，通过 `Fliter(..., rules=[...])` 指定顺序，`rules=[]` 不执行过滤规则。规则应保持返回文件与处理标识映射一致，且不修改输入。

入库时处理标识与 `ingestion_id`、文件内片段编号和总数一起写入 Qdrant payload，无需额外数据库。过滤时每页读取 256 条当前集合的完整性元数据，不读取正文或向量；按来源、处理标识、写入批次和预期片段数分组，只认可具有全部唯一编号的完整文件。重复记录不会补足缺失编号，不同写入批次或来源的残片不会被拼成完整文档。模型调用失败不记录处理标识；部分写入允许重试，但不自动清理原有残留记录。集合删除后处理信息随记录删除，各集合互不影响。

旧版本的 `document_hashes` / `document_fingerprint` 已替换为 `processing_keys` / `processing_key`。只有切分结果的旧记录无法准确还原原始文件哈希，因此不会直接用来跳过文件：首次构建会重新处理并追加新记录，之后才可在切分前跳过。已有历史记录不会删除。

构建应串行执行，且源文件在过滤到入库期间必须保持不变；当前不提供文件内容快照，读取检查与写入也不构成跨进程事务。没有可入库正文的文件不写完成记录，下次仍会检查、切分。每次过滤仍需扫描当前集合的完整性元数据，后续数据量增大时可再添加独立索引。

```bash
uv run python -m unittest discover -s tests -p test_fliter.py -v
```

## 知识库读取与 MCP

### 直接使用 Python

```python
from rag_mcp.mcp import KnowledgeBase

kb = KnowledgeBase(db_path="./.qdrant", collection_name="knowledge")
print(kb.stats())
print(kb.search("如何创建虚拟环境？", limit=5, min_score=0.3))
print(kb.keyword_search("ERROR42", limit=20))
print(kb.list_documents(limit=50))
```

`KnowledgeBase` 每次操作按需打开并关闭数据库，同一实例通过锁串行访问；不存在的数据库按空库返回，不创建存储目录。不扫描或读取原始文档，因此源文件移动或删除后仍可检索已入库内容。语义搜索仅为问题调用一次 Ollama，空库或没有匹配来源时不调用模型。

### 启动服务

```bash
uv run rag-mcp --db-path ./.qdrant --collection knowledge
# 等价模块入口：
uv run python -m rag_mcp.mcp.server --db-path ./.qdrant --collection knowledge
```

使用 stdio 传输，stdout 仅用于 MCP 协议。默认存储目录和集合与 `build_kb` 一致；自定义时需在构建端和服务端传入相同参数。服务工厂 `create_server(db_path, collection_name)` 只注册工具，导入模块不会扫描文件、打开数据库或请求模型。

MCP 客户端配置示例（把绝对路径替换为实际路径）：

```json
{
  "mcpServers": {
    "knowledge": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/rag-mcp",
        "run", "rag-mcp",
        "--db-path", "/absolute/path/to/rag-mcp/.qdrant",
        "--collection", "knowledge"
      ]
    }
  }
}
```

### 可用工具

所有工具只读，以 JSON 文本返回数据，不暴露向量，不提供删除、任意文件读取或由模型指定目录入库的工具。

| 工具 | 参数与用途 |
|---|---|
| `search_knowledge` | `text`、`limit=5`、`source=None`、`min_score=None`；Cosine 语义搜索，得分降序，最低分包含端点 |
| `search_keyword` | `text`、`limit=20`、`offset=0`、`source=None`；标题/正文大小写不敏感的子串匹配，适合函数名和错误码，不按相关度排序 |
| `list_documents` | `limit=50`、`offset=0`；列出来源、处理标识、写入批次、已存片段数和完整性 |
| `get_document` | `source`、`processing_key=None`、`ingestion_id=None`、`limit=20`、`offset=0`；读取已入库片段，不读取磁盘原文件 |
| `get_chunk` | `chunk_id`；根据检索返回的 `id` 精确读取一个片段，不存在返回 `null` |
| `get_knowledge_base_stats` | 无参数；集合是否存在、模型、维度、片段数、来源数及文档条目完整性 |

工具中的 `limit` 为 1–100；`offset` 是非负的**结果跳过条数**，不是 Qdrant 内部的 Point ID 游标。分页结果包含 `items`、`offset`、`limit`、`has_more`；文档列表和文档读取另有 `total`。关键词搜索不返回 `total`，找到足够匹配后即可停止，不会为统计总数额外扫描剩余记录。

搜索和片段读取结果包含 `id`、`text`、`source`、`heading_path`、原文定位及处理标识等字段；语义搜索另有 `score`。原始行号和字符偏移均从 0 开始、结束位置不包含。后续可用结果中的 `source` 读取文档，用 `id` 读取具体片段。

文档条目按 `(source, processing_key, ingestion_id)` 分组，同一来源的历史版本和不同写入尝试分别列出。`complete=true` 表示该次写入的片段编号完整，`false` 表示不完整，缺少完成度元数据的旧记录为 `null`。统计中的 `source_count` 是不同来源数，`document_count` 是这些分组的数量，并非唯一文件内容数。`get_document` 不指定版本和批次时返回该来源的所有记录，不自动推断“最新版”；返回的是有序 chunks，而非无损重建的原始文件。

目前语义搜索也可能返回历史版本和未完成写入留下的片段；需要指定版本时，可从文档列表取得处理标识和写入批次后调用 `get_document`。文档列表/统计需要扫描集合状态元数据，关键词搜索需要扫描候选正文，文档读取需收集所选来源的片段后排序，暂未实现独立文档索引或全文索引。读工具并不意味着全流程使用常量内存。

### 清理与迁移

MCP 相关实现统一位于 `mcp/`：服务入口是 `mcp/server.py`，工具位于 `mcp/tools/`，持久化读取逻辑位于 `mcp/knowledge_base.py`。原 `rag/` 和顶层 `tools/` 已迁移，Python 导入改为 `rag_mcp.mcp` 和 `rag_mcp.mcp.tools`，模块启动命令改为 `python -m rag_mcp.mcp.server`；CLI 命令 `rag-mcp` 保持不变。旧 `query.py`、`embeding_files.py` 内存构建/检索链路及 `embeding.py` 兼容入口已删除；对应测试由持久化检索、MCP 和 Ollama 测试替代。旧模块导入路径不再保留兼容层，搜索结果正文统一使用 `text` 字段而非旧 `content`。原始资料和已有数据库不删除。

测试使用临时真实 Qdrant 数据库，mock 模型请求，并包含真正的 stdio 子进程握手、工具注册及错误返回测试：

```bash
uv run python -m unittest discover -s tests -v
```
