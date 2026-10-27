"""保留原字符串向量化入口；实现集中在 embed 包中。"""

from .embed.ollama import MODEL, OLLAMA_URL, TIMEOUT_SECONDS
from .embed.ollama import embed_text as embeding

__all__ = ["MODEL", "OLLAMA_URL", "TIMEOUT_SECONDS", "embeding"]
