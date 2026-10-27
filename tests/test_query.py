import copy
import io
import math
import runpy
import unittest
import warnings
from contextlib import redirect_stdout
from unittest.mock import patch
from urllib.error import URLError

from rag_mcp.embeding_files import EmbeddedChunk
from rag_mcp.query import cosine_similarity, query


def make_chunk(name: str, vector: list[float]) -> EmbeddedChunk:
    return {
        "source": f"/{name}.md",
        "heading_path": [name],
        "content": f"{name} 的正文",
        "start_line": 1,
        "end_line": 2,
        "embedding": vector,
    }


class CosineSimilarityTests(unittest.TestCase):
    def test_same_direction(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 2.0], [2.0, 4.0]), 1.0)

    def test_perpendicular(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_direction(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_large_values_do_not_overflow_dot_product(self):
        self.assertAlmostEqual(cosine_similarity([1e200], [1e200]), 1.0)

    def test_invalid_vectors(self):
        for a, b in (
            ([], []),
            ([1.0], [1.0, 2.0]),
            ([0.0], [1.0]),
            ([1.0], [0.0]),
            ([math.nan], [1.0]),
            ([1.0], [math.inf]),
        ):
            with self.subTest(a=a, b=b):
                with self.assertRaises(ValueError):
                    cosine_similarity(a, b)


class QueryTests(unittest.TestCase):
    @patch("rag_mcp.query.embeding", return_value=[1.0, 0.0])
    def test_returns_top_five_in_descending_similarity_order(self, embed):
        vectors = {
            "third": [3.0, 1.0],
            "last": [-1.0, 0.0],
            "first": [1.0, 0.0],
            "sixth": [0.0, 1.0],
            "fifth": [1.0, 1.0],
            "second": [4.0, 1.0],
            "fourth": [2.0, 1.0],
        }
        data = {name: make_chunk(name, vector) for name, vector in vectors.items()}
        original = copy.deepcopy(data)
        results = query("如何安装？", data)
        expected = [data[name] for name in ("first", "second", "third", "fourth", "fifth")]
        self.assertEqual(len(results), 5)
        expected_scores = [1.0, 4 / math.sqrt(17), 3 / math.sqrt(10), 2 / math.sqrt(5), 1 / math.sqrt(2)]
        for result, item, score in zip(results, expected, expected_scores):
            self.assertEqual(
                {key: value for key, value in result.items() if key != "score"},
                item,
            )
            self.assertAlmostEqual(result["score"], score)
            self.assertIsNot(result, item)
        self.assertEqual(data, original)
        self.assertTrue(all("score" not in item for item in data.values()))
        embed.assert_called_once_with(
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery: 如何安装？"
        )

    @patch("rag_mcp.query.embeding", return_value=[1.0, 0.0])
    def test_returns_all_when_fewer_than_five(self, embed):
        data = {
            "a": make_chunk("a", [0.0, 1.0]),
            "b": make_chunk("b", [1.0, 0.0]),
        }
        self.assertEqual(
            query("问题", data),
            [{**data["b"], "score": 1.0}, {**data["a"], "score": 0.0}],
        )

    @patch("rag_mcp.query.embeding", return_value=[1.0, 0.0])
    def test_returns_all_when_exactly_five(self, embed):
        data = {str(i): make_chunk(str(i), [1.0, float(i)]) for i in range(5)}
        results = query("问题", data)
        self.assertEqual(len(results), 5)
        self.assertEqual(
            [result["source"] for result in results],
            [item["source"] for item in data.values()],
        )
        for index, result in enumerate(results):
            self.assertAlmostEqual(result["score"], 1 / math.sqrt(1 + index * index))

    @patch("rag_mcp.query.embeding")
    def test_empty_data_does_not_request_embedding(self, embed):
        self.assertEqual(query("问题", {}), [])
        embed.assert_not_called()

    @patch("rag_mcp.query.embeding", return_value=[1.0, 0.0])
    def test_same_scores_preserve_input_order(self, embed):
        data = {str(i): make_chunk(str(i), [1.0, 0.0]) for i in range(7)}
        self.assertEqual(
            query("问题", data),
            [{**chunk, "score": 1.0} for chunk in list(data.values())[:5]],
        )

    @patch("rag_mcp.query.embeding", return_value=[1.0, 0.0])
    def test_uses_cosine_not_raw_dot_product(self, embed):
        data = {
            "large": make_chunk("large", [100.0, 200.0]),
            "close": make_chunk("close", [1.0, 1.0]),
        }
        results = query("问题", data)
        self.assertEqual(
            [result["source"] for result in results],
            [data["close"]["source"], data["large"]["source"]],
        )
        self.assertAlmostEqual(results[0]["score"], 1 / math.sqrt(2))
        self.assertAlmostEqual(results[1]["score"], 1 / math.sqrt(5))

    @patch("rag_mcp.query.embeding", return_value=[1.0, 0.0])
    def test_negative_similarity_is_preserved(self, embed):
        data = {"opposite": make_chunk("opposite", [-1.0, 0.0])}
        self.assertEqual(query("问题", data), [{**data["opposite"], "score": -1.0}])

    @patch("rag_mcp.query.embeding", side_effect=[[1.0, 0.0], [0.0, 1.0]])
    def test_scores_are_specific_to_each_query(self, embed):
        data = {
            "a": make_chunk("a", [1.0, 0.0]),
            "b": make_chunk("b", [0.0, 1.0]),
        }
        first = query("第一个问题", data)
        second = query("第二个问题", data)
        self.assertEqual(first, [{**data["a"], "score": 1.0}, {**data["b"], "score": 0.0}])
        self.assertEqual(second, [{**data["b"], "score": 1.0}, {**data["a"], "score": 0.0}])
        self.assertTrue(all("score" not in item for item in data.values()))

    def test_main_example_prints_score_and_content(self):
        data = {"demo": make_chunk("示例", [1.0, 0.0])}
        output = io.StringIO()
        with (
            patch("rag_mcp.embeding_files.embeding_files", return_value=data),
            patch("rag_mcp.embeding.embeding", return_value=[1.0, 0.0]),
            redirect_stdout(output),
            warnings.catch_warnings(),
        ):
            # 本测试文件已经导入 query；这里有意再次以 __main__ 执行示例。
            warnings.filterwarnings(
                "ignore",
                message="'rag_mcp.query' found in sys.modules",
                category=RuntimeWarning,
            )
            runpy.run_module("rag_mcp.query", run_name="__main__")
        self.assertIn("第 1 条 | 相似度：1.0000", output.getvalue())
        self.assertIn("示例 的正文", output.getvalue())

    @patch("rag_mcp.query.embeding")
    def test_invalid_text_fails_before_request(self, embed):
        for text, error in ((None, TypeError), (123, TypeError), ("", ValueError), (" \n", ValueError)):
            with self.subTest(text=text):
                with self.assertRaises(error):
                    query(text, {})
        embed.assert_not_called()

    @patch("rag_mcp.query.embeding", return_value=[1.0, 0.0])
    def test_incompatible_chunk_dimension_raises(self, embed):
        with self.assertRaises(ValueError):
            query("问题", {"a": make_chunk("a", [1.0])})

    @patch("rag_mcp.query.embeding", side_effect=URLError("无法连接 Ollama"))
    def test_model_request_error_propagates(self, embed):
        with self.assertRaises(URLError):
            query("问题", {"a": make_chunk("a", [1.0, 0.0])})


if __name__ == "__main__":
    unittest.main()
