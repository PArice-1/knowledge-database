import importlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import URLError

from rag_mcp.embeding_files import embeding_files


class EmbedingFilesTests(unittest.TestCase):
    @patch("rag_mcp.embed.embeder.embed_text", return_value=[0.25, -0.5])
    def test_packages_content_metadata_and_vector(self, embed):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "demo.md"
            file.write_text("# 指南\n## 安装\n正文。\n", encoding="utf-8")
            data = embeding_files(root)
            self.assertEqual(len(data), 1)
            self.assertEqual(
                data[f"{file.resolve()}:11:15"],
                {
                    "source": str(file.resolve()),
                    "heading_path": ["指南", "安装"],
                    "content": "正文。\n",
                    "start_line": 2,
                    "end_line": 3,
                    "embedding": [0.25, -0.5],
                },
            )
            embed.assert_called_once_with(text="指南 > 安装\n\n正文。\n")
            self.assertEqual(json.loads(json.dumps(data)), data)

    @patch("rag_mcp.embed.embeder.embed_text", side_effect=[[0.1], [0.2], [0.3]])
    def test_duplicate_headings_do_not_overwrite_chunks(self, embed):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.md").write_text(
                "# 同名标题\n第一段。\n# 同名标题\n第二段。\n", encoding="utf-8"
            )
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.md").write_text("# 同名标题\n第三段。\n", encoding="utf-8")
            data = embeding_files(root)
            self.assertEqual(len(data), 3)
            self.assertEqual(
                [entry["content"] for entry in data.values()],
                ["第一段。\n", "第二段。\n", "第三段。\n"],
            )
            self.assertEqual(
                [entry["embedding"] for entry in data.values()], [[0.1], [0.2], [0.3]]
            )
            self.assertEqual(embed.call_count, 3)

    @patch("rag_mcp.embed.embeder.embed_text", return_value=[0.1])
    def test_txt_is_embedded_without_heading_context(self, embed):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.TXT").write_text("# 普通文本\n正文。", encoding="utf-8")
            (root / "ignored.pdf").touch()
            data = embeding_files(root)
            self.assertEqual(len(data), 1)
            chunk = next(iter(data.values()))
            self.assertEqual(chunk["heading_path"], [])
            self.assertEqual(chunk["content"], "# 普通文本\n正文。")
            embed.assert_called_once_with(text="# 普通文本\n正文。")

    @patch("rag_mcp.embed.embeder.embed_text", return_value=[0.1])
    def test_same_line_subchunks_do_not_overwrite_each_other(self, embed):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            body = "重复正文" * 1500
            for name in ("long.md", "long.txt"):
                with self.subTest(name=name):
                    file = root / name
                    file.write_text(body, encoding="utf-8")
                    data = embeding_files(root)
                    self.assertEqual(len(data), 3)
                    self.assertEqual("".join(item["content"] for item in data.values()), body)
                    self.assertTrue(
                        all((item["start_line"], item["end_line"]) == (0, 1) for item in data.values())
                    )
                    self.assertEqual(
                        list(data),
                        [f"{file.resolve()}:0:2048", f"{file.resolve()}:2048:4096", f"{file.resolve()}:4096:6000"],
                    )
                    file.unlink()

    @patch("rag_mcp.embed.embeder.embed_text", return_value=[0.1])
    def test_headingless_document_uses_body_as_input(self, embed):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plain.md").write_text("没有标题的正文。", encoding="utf-8")
            data = embeding_files(root)
            self.assertEqual(next(iter(data.values()))["heading_path"], [])
            embed.assert_called_once_with(text="没有标题的正文。")

    @patch("rag_mcp.embed.embeder.embed_text")
    def test_empty_knowledge_base_makes_no_model_requests(self, embed):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(embeding_files(root), {})
            (root / "empty.md").touch()
            self.assertEqual(embeding_files(root), {})
            embed.assert_not_called()

    @patch("rag_mcp.embed.embeder.embed_text")
    def test_missing_directory_raises_before_model_request(self, embed):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                embeding_files(Path(directory) / "missing")
            embed.assert_not_called()

    @patch("rag_mcp.embed.embeder.embed_text", side_effect=URLError("无法连接 Ollama"))
    def test_model_error_propagates(self, embed):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "demo.md").write_text("正文。", encoding="utf-8")
            with self.assertRaises(URLError):
                embeding_files(root)

    def test_import_does_not_scan_files_or_call_model(self):
        module = importlib.import_module("rag_mcp.embeding_files")
        try:
            with (
                patch("rag_mcp.chunk.search_files") as search,
                patch("rag_mcp.embed.embeder.embed_text") as embed,
            ):
                importlib.reload(module)
                search.assert_not_called()
                embed.assert_not_called()
        finally:
            # 离开 patch 后再次加载，避免模块保存被 mock 的导入。
            importlib.reload(module)


if __name__ == "__main__":
    unittest.main()
