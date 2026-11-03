import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from rag_mcp.embed.ollama import MODEL, OLLAMA_URL, TIMEOUT_SECONDS
from rag_mcp.embed.ollama import embed_text as embeding


class EmbedingTests(unittest.TestCase):
    def response(self, payload):
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    @patch("rag_mcp.embed.ollama.urlopen")
    def test_returns_one_dimensional_float_vector(self, urlopen):
        urlopen.return_value = self.response({"embeddings": [[0.25, -0.5, 1]]})
        vector = embeding("安装 Python")
        self.assertEqual(vector, [0.25, -0.5, 1.0])
        self.assertTrue(all(isinstance(value, float) for value in vector))

    @patch("rag_mcp.embed.ollama.urlopen")
    def test_request_preserves_input_and_disables_truncation(self, urlopen):
        urlopen.return_value = self.response({"embeddings": [[0.1]]})
        text = "  安装指南\n\n正文。\n"
        embeding(text)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, OLLAMA_URL)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(
            json.loads(request.data),
            {"model": MODEL, "input": text, "truncate": False},
        )
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": TIMEOUT_SECONDS})

    @patch("rag_mcp.embed.ollama.urlopen")
    def test_rejects_non_string_input_before_request(self, urlopen):
        for text in (None, 42, ["正文"]):
            with self.subTest(text=text), self.assertRaises(TypeError):
                embeding(text)
        urlopen.assert_not_called()

    @patch("rag_mcp.embed.ollama.urlopen")
    def test_rejects_blank_input_before_request(self, urlopen):
        for text in ("", " \n\t"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                embeding(text)
        urlopen.assert_not_called()

    @patch("rag_mcp.embed.ollama.urlopen")
    def test_rejects_invalid_response_vectors(self, urlopen):
        for payload in (
            None,
            [],
            {},
            {"embeddings": None},
            {"embeddings": []},
            {"embeddings": [[0.1], [0.2]]},
            {"embeddings": [[]]},
            {"embeddings": ["not a vector"]},
            {"embeddings": [["0.1"]]},
            {"embeddings": [[True]]},
            {"embeddings": [[None]]},
            {"embeddings": [[[0.1]]]},
            {"embeddings": [[float("nan")]]},
            {"embeddings": [[float("inf")]]},
        ):
            with self.subTest(payload=payload):
                urlopen.return_value = self.response(payload)
                with self.assertRaises(ValueError):
                    embeding("正文")

    @patch("rag_mcp.embed.ollama.urlopen")
    def test_network_errors_propagate(self, urlopen):
        for error in (
            URLError("服务未启动"),
            HTTPError(OLLAMA_URL, 404, "模型未找到", None, None),
            TimeoutError("请求超时"),
        ):
            with self.subTest(error=error):
                urlopen.side_effect = error
                with self.assertRaises(type(error)):
                    embeding("正文")

    @patch("rag_mcp.embed.ollama.urlopen")
    def test_invalid_json_propagates(self, urlopen):
        urlopen.return_value = io.BytesIO(b"not json")
        with self.assertRaises(json.JSONDecodeError):
            embeding("正文")


if __name__ == "__main__":
    unittest.main()
