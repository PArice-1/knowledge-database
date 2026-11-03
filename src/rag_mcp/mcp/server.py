"""基于已构建的 Qdrant 知识库提供只读 MCP 工具。"""

import argparse
from pathlib import Path

from mcp.server import MCPServer

from ..config import DEFAULT_COLLECTION, DEFAULT_DB_PATH
from .tools import register_tools
from .knowledge_base import KnowledgeBase


def create_server(
    db_path: str | Path = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> MCPServer:
    """创建工具注册表；不扫描源文件、不请求模型、不打开数据库。"""
    server = MCPServer("rag-mcp")
    register_tools(server, KnowledgeBase(db_path, collection_name))
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 stdio 提供持久化知识库 MCP 工具")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="Qdrant 存储目录")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="集合名称")
    args = parser.parse_args()
    create_server(args.db_path, args.collection).run(transport="stdio")


if __name__ == "__main__":
    main()
