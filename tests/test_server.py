import unittest
from unittest.mock import patch

from rag_mcp.server import mcp


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_tool_is_registered_with_string_parameter(self):
        tools = await mcp.list_tools()
        tool = next(tool for tool in tools if tool.name == "search_knowledge")
        self.assertEqual(tool.input_schema["properties"]["text"]["type"], "string")
        self.assertIn("text", tool.input_schema["required"])
        self.assertIn("知识库", tool.description)

    @patch("rag_mcp.server.tool_query", return_value='[{"content":"测试正文","score":0.9}]')
    async def test_mcp_tool_returns_query_json_as_text(self, tool_query):
        result = await mcp.call_tool("search_knowledge", {"text": "如何安装？"})
        tool_query.assert_called_once_with(text="如何安装？")
        self.assertEqual(len(result.content), 1)
        self.assertEqual(result.content[0].type, "text")
        self.assertEqual(result.content[0].text, tool_query.return_value)


if __name__ == "__main__":
    unittest.main()
