import asyncio
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mcp import Client, StdioServerParameters, stdio_client

from rag_mcp import main as package_main
from rag_mcp.build_kb import build_kb
from rag_mcp.rag.server import create_server, main


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_six_read_only_tools_are_registered_with_expected_schemas(self):
        server = create_server()
        tools = {tool.name: tool for tool in await server.list_tools()}
        self.assertEqual(set(tools), {
            "search_knowledge", "search_keyword", "list_documents", "get_document", "get_chunk",
            "get_knowledge_base_stats",
        })
        search = tools["search_knowledge"]
        self.assertEqual(search.input_schema["properties"]["text"]["type"], "string")
        self.assertIn("text", search.input_schema["required"])
        self.assertEqual(search.input_schema["properties"]["limit"]["default"], 5)
        for tool in tools.values():
            self.assertTrue(tool.annotations.read_only_hint)
            self.assertFalse(tool.annotations.destructive_hint)
            self.assertTrue(tool.description)

    async def test_server_creation_has_no_filesystem_or_model_side_effects(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "not-created"
            with (
                patch("rag_mcp.chunk.Chunker.chunk", side_effect=AssertionError("不能切分")),
                patch("rag_mcp.embed.ollama.embed_text", side_effect=AssertionError("不能请求模型")),
                patch("rag_mcp.rag.knowledge_base.VectorDatabase", side_effect=AssertionError("不能打开数据库")),
            ):
                server = create_server(path)
                self.assertEqual(len(await server.list_tools()), 6)
            self.assertFalse(path.exists())

    async def test_build_multiple_roots_then_search_and_read_via_mcp(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "one", root / "two"
            first.mkdir()
            second.mkdir()
            (first / "guide.md").write_text("# 安装\n运行 Python 创建环境。", encoding="utf-8")
            (second / "notes.txt").write_text("错误码 ERROR42 的处理方法。", encoding="utf-8")
            db_path = root / "db"
            with patch("rag_mcp.embed.embeder.embed_text", return_value=[1.0, 0.0, 0.0]) as embed:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(build_kb([str(first), second], db_path=db_path, collection_name="docs"), 2)
                self.assertEqual(embed.call_count, 2)
            server = create_server(db_path, "docs")
            with (
                patch("rag_mcp.chunk.Chunker.chunk", side_effect=AssertionError("搜索不应切分")),
                patch("rag_mcp.embed.ollama.embed_text", return_value=[1.0, 0.0, 0.0]) as query_embed,
            ):
                result = await server.call_tool("search_knowledge", {"text": "创建环境", "source": str(first / "guide.md")})
                self.assertFalse(result.is_error)
                found = json.loads(result.content[0].text)
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]["text"], "运行 Python 创建环境。")
                chunk = await server.call_tool("get_chunk", {"chunk_id": found[0]["id"]})
                self.assertEqual(json.loads(chunk.content[0].text)["text"], found[0]["text"])
                keyword = await server.call_tool("search_keyword", {"text": "ERROR42"})
                self.assertEqual(len(json.loads(keyword.content[0].text)["items"]), 1)
                stats = await server.call_tool("get_knowledge_base_stats", {})
                self.assertEqual(json.loads(stats.content[0].text)["chunk_count"], 2)
                self.assertEqual(query_embed.call_count, 1)
                # 同一服务的工具可能在线程池中并发运行，数据库访问必须串行打开/关闭。
                parallel = await asyncio.gather(*(
                    server.call_tool("get_knowledge_base_stats", {}) for _ in range(4)
                ))
                self.assertTrue(all(not result.is_error for result in parallel))

    async def test_missing_database_returns_empty_results_not_protocol_errors(self):
        with TemporaryDirectory() as directory:
            server = create_server(Path(directory) / "missing")
            result = await server.call_tool("search_knowledge", {"text": "问题"})
            self.assertFalse(result.is_error)
            self.assertEqual(json.loads(result.content[0].text), [])

    async def test_real_stdio_server_handshake_tools_and_error_result(self):
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "missing"
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "rag_mcp.rag.server", "--db-path", str(db_path)],
            )
            async with asyncio.timeout(30), Client(stdio_client(params), mode="legacy") as client:
                tools = await client.list_tools()
                self.assertEqual(len(tools.tools), 6)
                stats = await client.call_tool("get_knowledge_base_stats", {})
                self.assertFalse(stats.is_error)
                self.assertFalse(json.loads(stats.content[0].text)["exists"])
                invalid = await client.call_tool("search_knowledge", {"text": "问题", "limit": 0})
                self.assertTrue(invalid.is_error)
                self.assertIn("limit", invalid.content[0].text)
                self.assertFalse(db_path.exists())

    @patch("rag_mcp.rag.server.create_server")
    def test_server_cli_passes_storage_and_collection_and_uses_stdio(self, create):
        with patch("sys.argv", ["rag-mcp", "--db-path", "./db", "--collection", "docs"]):
            main()
        create.assert_called_once_with("./db", "docs")
        create.return_value.run.assert_called_once_with(transport="stdio")

    @patch("rag_mcp.rag.server.main")
    def test_package_entrypoint_starts_server(self, serve):
        package_main()
        serve.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
