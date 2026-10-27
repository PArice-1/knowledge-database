import copy
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from rag_mcp.tools import tool_query


class ToolQueryTests(unittest.TestCase):
    @patch("rag_mcp.tools.query")
    @patch("rag_mcp.tools.embeding_files")
    def test_returns_json_without_embedding_and_preserves_other_fields(self, build, search):
        knowledges = {"demo": {"content": "原文"}}
        results = [
            {
                "source": "/资料/demo.md",
                "heading_path": ["安装指南", "Linux"],
                "content": '运行 "python"。\n保留换行和反斜杠：\\',
                "start_line": 1,
                "end_line": 3,
                "embedding": [0.1, -0.2],
                "score": 0.85,
            }
        ]
        original = copy.deepcopy(results)
        build.return_value = knowledges
        search.return_value = results

        output = tool_query("如何安装？")

        self.assertIsInstance(output, str)
        decoded = json.loads(output)
        self.assertEqual(
            decoded,
            [{key: value for key, value in results[0].items() if key != "embedding"}],
        )
        self.assertNotIn("embedding", decoded[0])
        self.assertIn("安装指南", output)
        self.assertEqual(results, original)
        build.assert_called_once_with()
        search.assert_called_once_with("如何安装？", knowledges)

    @patch("rag_mcp.tools.query")
    @patch("rag_mcp.tools.embeding_files", return_value={})
    def test_filters_each_result_without_changing_order_or_input(self, build, search):
        results = [
            {"content": "第一条", "score": 0.9, "embedding": [0.1]},
            {"content": "第二条", "score": 0.8, "embedding": [0.2]},
        ]
        original = copy.deepcopy(results)
        search.return_value = results
        self.assertEqual(
            json.loads(tool_query("问题")),
            [{"content": "第一条", "score": 0.9}, {"content": "第二条", "score": 0.8}],
        )
        self.assertEqual(results, original)

    @patch("rag_mcp.tools.query", return_value=[])
    @patch("rag_mcp.tools.embeding_files", return_value={})
    def test_empty_results_return_json_array(self, build, search):
        self.assertEqual(tool_query("问题"), "[]")

    @patch("rag_mcp.tools.query")
    @patch("rag_mcp.tools.embeding_files", side_effect=URLError("模型不可用"))
    def test_build_error_propagates(self, build, search):
        with self.assertRaises(URLError):
            tool_query("问题")
        search.assert_not_called()

    @patch("rag_mcp.tools.query", side_effect=ValueError("查询失败"))
    @patch("rag_mcp.tools.embeding_files", return_value={})
    def test_query_error_propagates(self, build, search):
        with self.assertRaisesRegex(ValueError, "查询失败"):
            tool_query("问题")

    @patch("rag_mcp.tools.query", return_value=[{"score": float("nan")}])
    @patch("rag_mcp.tools.embeding_files", return_value={})
    def test_non_finite_score_is_not_serialized_as_invalid_json(self, build, search):
        with self.assertRaises(ValueError):
            tool_query("问题")


if __name__ == "__main__":
    unittest.main()
