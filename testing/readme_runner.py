"""Run the Python examples of a package README against the offline stub.

Every ``lexapi.LexAPI`` / ``AsyncLexAPI`` an example builds (directly, or
through a package that builds its own clients) answers from synthetic
fixtures on a loopback base URL. ``my_llm`` is a mock LLM.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import io
import pathlib
import re
import sys
import types
from typing import Any

import httpx
import lexapi
import pytest
from lexapi_offline import LOOPBACK, OFFLINE_KEY, Recorder, Route
from llama_index.core.llms import MockLLM

_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def python_blocks(readme: pathlib.Path) -> list[str]:
    return _BLOCK.findall(readme.read_text(encoding="utf-8"))


def offline_classes(
    routes: dict[str, Route],
) -> tuple[type[lexapi.LexAPI], type[lexapi.AsyncLexAPI]]:
    recorder = Recorder(routes)

    async def answer_async(request: httpx.Request) -> httpx.Response:
        return recorder.answer(request)

    class OfflineLexAPI(lexapi.LexAPI):
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            super().__init__(
                api_key or OFFLINE_KEY,
                base_url=LOOPBACK,
                transport=httpx.MockTransport(recorder.answer),
                max_retries=0,
                user_agent_suffix=kwargs.get("user_agent_suffix"),
            )

    class OfflineAsyncLexAPI(lexapi.AsyncLexAPI):
        def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
            super().__init__(
                api_key or OFFLINE_KEY,
                base_url=LOOPBACK,
                transport=httpx.MockTransport(answer_async),
                max_retries=0,
                user_agent_suffix=kwargs.get("user_agent_suffix"),
            )

    return OfflineLexAPI, OfflineAsyncLexAPI


def run_readme(
    readme: pathlib.Path,
    routes: dict[str, Route],
    client_module: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Execute every Python block of ``readme`` in one namespace; return the
    namespace and what the examples printed."""
    sync_cls, async_cls = offline_classes(routes)
    monkeypatch.setattr(lexapi, "LexAPI", sync_cls)
    monkeypatch.setattr(lexapi, "AsyncLexAPI", async_cls)
    monkeypatch.setattr(client_module, "LexAPI", sync_cls)
    monkeypatch.setattr(client_module, "AsyncLexAPI", async_cls)
    namespace: dict[str, Any] = {"my_llm": MockLLM(), **(extra or {})}
    out = io.StringIO()
    blocks = python_blocks(readme)
    assert blocks, f"no python blocks in {readme}"
    with contextlib.redirect_stdout(out):
        for block in blocks:
            code = compile(block, str(readme), "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            result = eval(code, namespace)
            if asyncio.iscoroutine(result):
                asyncio.run(result)
    sys.stdout.flush()
    return namespace, out.getvalue()
