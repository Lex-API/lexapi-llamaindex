"""Offline test harness shared by both packages' test suites.

Nothing here is shipped: it is imported only by ``tests/conftest.py`` of each
package, which puts this directory on ``sys.path``. Neither distribution
depends on the other.

- Clients answer from ``httpx.MockTransport`` with fixtures from
  ``fixtures/``; their base URL must start with ``http://127.0.0.1:``.
- :func:`network_guard` makes the real httpx transports raise, so a client
  built without a mock transport (for example with the SDK's default base
  URL, which is the real API) cannot send anything.
- Every fixture is synthetic.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from lexapi import AsyncLexAPI, LexAPI

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
LOOPBACK = "http://127.0.0.1:9/api/v1"
OFFLINE_KEY = "lex_offline_test_key_0123456789"
BLOCKED = "Offline tests blocked a real HTTP request"

#: Environment variables removed for the whole test session.
SCRUBBED_ENV = (
    "LEXAPI_API_KEY",
    "LEXAPI_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


def load_fixture(name: str) -> Any:
    """The decoded body of ``fixtures/<name>``."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def assert_offline_base_url(base_url: str) -> None:
    """Refuse any base URL that is not loopback."""
    if not base_url.startswith("http://127.0.0.1:"):
        raise RuntimeError(f"Offline tests refuse the base URL {base_url!r}")


def assert_offline_client(client: LexAPI | AsyncLexAPI) -> None:
    """Refuse a client whose base URL is not loopback."""
    assert_offline_base_url(str(client._http.base_url))


@dataclass
class Route:
    """One canned answer: HTTP status, JSON body (a ``str`` is sent as the
    raw body, e.g. a truncated one) and extra headers."""

    body: Any
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)


def fixture_route(name: str, status: int = 200, headers: dict[str, str] | None = None) -> Route:
    return Route(load_fixture(name), status, dict(headers or {}))


@dataclass
class Recorder:
    """Records every request and answers from ``routes`` (keyed by URL path
    below the base URL, e.g. ``/legislation/semantic``)."""

    routes: dict[str, Route]
    requests: list[httpx.Request] = field(default_factory=list)

    def answer(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != "127.0.0.1":
            raise RuntimeError(f"Offline tests refuse the host {request.url.host!r}")
        self.requests.append(request)
        path = request.url.path.removeprefix("/api/v1")
        route = self.routes.get(path)
        if route is None:
            return httpx.Response(
                404, json={"success": False, "error": {"code": "NOT_FOUND", "message": path}}
            )
        if isinstance(route.body, str):
            headers = {"content-type": "application/json", **route.headers}
            return httpx.Response(route.status, content=route.body.encode(), headers=headers)
        return httpx.Response(route.status, json=route.body, headers=route.headers)

    def bodies(self) -> list[Any]:
        return [json.loads(r.content) if r.content else None for r in self.requests]


def make_client(routes: dict[str, Route], base_url: str = LOOPBACK) -> tuple[LexAPI, Recorder]:
    assert_offline_base_url(base_url)
    recorder = Recorder(routes)
    client = LexAPI(
        OFFLINE_KEY,
        base_url=base_url,
        transport=httpx.MockTransport(recorder.answer),
        max_retries=0,
    )
    assert_offline_client(client)
    return client, recorder


def make_async_client(
    routes: dict[str, Route], base_url: str = LOOPBACK
) -> tuple[AsyncLexAPI, Recorder]:
    assert_offline_base_url(base_url)
    recorder = Recorder(routes)

    async def answer(request: httpx.Request) -> httpx.Response:
        return recorder.answer(request)

    client = AsyncLexAPI(
        OFFLINE_KEY,
        base_url=base_url,
        transport=httpx.MockTransport(answer),
        max_retries=0,
    )
    assert_offline_client(client)
    return client, recorder


def _refuse(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(BLOCKED)


async def _refuse_async(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(BLOCKED)


@contextmanager
def network_guard() -> Iterator[None]:
    """Patch the real httpx transports to raise and scrub API keys from the
    environment for as long as the context is open."""
    saved_env = {name: os.environ.pop(name) for name in SCRUBBED_ENV if name in os.environ}
    sync_original: Callable[..., Any] = httpx.HTTPTransport.handle_request
    async_original: Callable[..., Any] = httpx.AsyncHTTPTransport.handle_async_request
    httpx.HTTPTransport.handle_request = _refuse  # type: ignore[method-assign]
    httpx.AsyncHTTPTransport.handle_async_request = _refuse_async  # type: ignore[method-assign]
    try:
        yield
    finally:
        httpx.HTTPTransport.handle_request = sync_original  # type: ignore[method-assign]
        httpx.AsyncHTTPTransport.handle_async_request = async_original  # type: ignore[method-assign]
        os.environ.update(saved_env)


def disable_llm_and_embeddings() -> None:
    """Point LlamaIndex's global settings at mock models, so nothing can
    reach an LLM or embedding provider."""
    from llama_index.core import MockEmbedding, Settings
    from llama_index.core.llms import MockLLM

    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=8)


# ── synthetic body generators (property tests) ─────────────────────────

SYN = "[SYNTHETIC FIXTURE - not the text of the act] "


def legislation_hit(
    n: int,
    *,
    resolved: bool | None,
    consolidated: bool = False,
    reason: str = "fetch_timeout",
    as_of: str = "2099-06-01",
    echoed_as_of: str | None = None,
    with_text: bool = True,
) -> dict[str, Any]:
    """A legislation hit whose text and snippet carry unique tokens
    (``TXT<n>Q``, ``SNP<n>Q``) so a leak can be searched for."""
    celex = f"0209{n % 10}R{n:04d}-20990101" if consolidated else f"3209{n % 10}R{n:04d}"
    hit: dict[str, Any] = {
        "id": f"h-{n}",
        "law_id": celex,
        "law_title": f"Synthetic act {n}",
        "article_ref": f"art_{n}",
        "language": "en",
        "score": round(0.5 + (n % 50) / 100, 2),
        "snippet": f"{SYN}SNP{n}Q",
    }
    if with_text:
        hit["text"] = f"{SYN}TXT{n}Q"
    if consolidated:
        hit.update(
            {
                "consolidated": True,
                "consolidatedAsOf": "2099-01-01",
                "baseCelex": f"3209{n % 10}R{n:04d}",
                "consolidatedNote": "Consolidated text; not the authentic text.",
            }
        )
    if resolved is not None:
        hit["asOf"] = echoed_as_of or as_of
        hit["asOfResolved"] = resolved
        if resolved:
            hit["asOfVersion"] = {
                "celex": celex,
                "date": "2099-01-01" if consolidated else None,
                "kind": "consolidated" if consolidated else "as_published",
            }
        else:
            hit["asOfReason"] = reason
    return hit


def legislation_body(hits: list[dict[str, Any]], as_of: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "success": True,
        "searchType": "semantic-legislation",
        "query": "synthetic",
        "resultCount": len(hits),
        "results": hits,
        "credits": {"operation_weight": 5, "remaining": 100},
    }
    if as_of is not None:
        body["asOf"] = as_of
    return body
