# rag-mcp

本地持久化 RAG 知识库：将 Markdown/TXT 资料写入 Qdrant，通过 MCP 为 Agent 提供语义搜索、关键词搜索和文档读取。

- 支持 UTF-8 编码的 `.md`、`.txt`，可导入多个目录或文件。
- 使用 Ollama 生成向量，Qdrant 本地存储，无需单独部署数据库服务。
- 支持启动前增量构建，未变化且完整入库的文件跳过切分和向量化。
- 提供六个只读 MCP 工具；最终答案由接入的 Agent 生成，本项目不包含聊天界面。

## 快速开始

需要 Python 3.12+、[uv](https://docs.astral.sh/uv/) 和 [Ollama](https://ollama.com/)。以下命令在项目根目录执行。

### 1. 安装依赖并准备模型

```bash
uv sync --frozen
ollama pull qwen3-embedding:0.6b
```

确保 Ollama 服务已运行；如果尚未启动，先在另一个终端执行 `ollama serve`。默认向量接口为 `http://localhost:11434/api/embed`。

### 2. 准备资料并启动

```bash
mkdir -p knowledge-base
# 将需要导入的 MD/TXT 资料放入 knowledge-base 后执行：
uv run rag-mcp --build-on-start
```

默认扫描 `./knowledge-base`，数据库保存到 `./knowledge-base/storage`，集合为 `knowledge`。构建完成后通过 stdio 等待 MCP 客户端连接，不会出现交互式问答提示。

也可以先单独构建，再启动只读服务：

```bash
uv run rag-build-kb ./knowledge-base
uv run rag-mcp
```

自定义多个资料路径和存储位置：

```bash
uv run rag-mcp --build-on-start --source /path/to/docs /path/to/notes.md \
  --db-path /path/to/storage --collection knowledge
```

示例路径需替换为实际路径。更多构建参数见[构建与去重](docs/ingestion.md)，启动参数见 [MCP 与检索](docs/mcp.md)。

## 接入 Agent

在支持 `mcpServers` 配置格式的客户端中添加以下配置，将 `/absolute/path/to/rag-mcp` 替换为项目的实际绝对路径：

```json
{
  "mcpServers": {
    "knowledge": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/rag-mcp",
        "run", "rag-mcp", "--build-on-start"
      ]
    }
  }
}
```

由客户端启动服务即可，无需再手动启动一份。连接后可以让 Agent：

> 先搜索我的知识库，说明项目如何安装，并注明资料来源。

可用工具：`search_knowledge`、`search_keyword`、`list_documents`、`get_document`、`get_chunk`、`get_knowledge_base_stats`。参数和返回结构见 [MCP 与检索](docs/mcp.md)。

## 使用注意

- 启动构建只检查一次，不监听运行期间的文件变化；不加 `--build-on-start` 则不检查源文件。
- 修改文件会追加新版本，不会自动删除旧版本或已删除文件的记录，搜索也不自动限定最新版。
- 语义搜索只向量化当前问题；其余工具不调用 Ollama。启动构建的普通进度输出静默，异常仍正常抛出。
- 不要让多个 MCP 服务或构建进程同时操作同一个本地数据库。首次构建较慢时，先单独构建或调大客户端启动超时。

## 文档

| 文档 | 内容 |
|---|---|
| [架构总览](docs/architecture.md) | 构建与检索流程、模块职责、目录结构和迁移说明 |
| [构建与去重](docs/ingestion.md) | CLI/Python 构建、处理标识、完整性判断和重试限制 |
| [文件切分](docs/chunking.md) | Markdown/TXT 规则、片段结构、行号和字符偏移 |
| [片段向量化](docs/embeddings.md) | Ollama 配置、输入组织和向量结果结构 |
| [向量存储](docs/storage.md) | Qdrant 集合、写入、持久化和客户端生命周期 |
| [MCP 与检索](docs/mcp.md) | 启动配置、六个工具、Python 检索与分页语义 |

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

测试使用临时数据库并 mock 模型请求，无需运行 Ollama。
