"""工具公共的 JSON 序列化和只读声明。"""

import json
from typing import Any

from mcp.types import ToolAnnotations

READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False,
)


def as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
