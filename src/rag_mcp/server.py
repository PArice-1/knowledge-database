"""使用 MCP Python SDK 2.x 暴露本地知识库搜索工具。"""

from mcp.server import MCPServer

from .tools import tool_query

mcp = MCPServer("rag-mcp")

@mcp.tool()
def search_knowledge(text: str) -> str:
    """搜索本地 Markdown/TXT 知识库, 返回最多五个相关片段及相似度、信息来源"""
    return tool_query(text=text)


def start_server():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    mcp.run(transport="stdio")
