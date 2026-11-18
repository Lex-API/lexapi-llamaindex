"""LexAPI tool spec: search EU legislation, read a document, and the version
of an act for a date.

Every tool converts the API's answer with :mod:`lexapi.passages` and returns
one string for the model: each passage as :func:`lexapi.passages.format_passage`
(the text or a notice, then a ``Source:`` line with the citation).

Annotations here are evaluated at runtime on purpose (no ``from __future__
import annotations``): LlamaIndex builds each tool's argument schema from
the function signature.
"""

import asyncio
import inspect
import re
from collections.abc import Callable
from typing import Any, TypeVar, get_args

import llama_index.core.instrumentation as instrument
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.tools import FunctionTool, ToolMetadata
from llama_index.core.tools.tool_spec.base import SPEC_FUNCTION_TYPE, BaseToolSpec

from lexapi import AsyncLexAPI, LexAPI, LexAPIError, passages
from lexapi.models import DocumentResponse, LegislationAsOfResponse, SemanticLegislationResponse
from lexapi.passages import PassageResult, UnresolvedPolicy
from llama_index.tools.lexapi._client import Clients

dispatcher = instrument.get_dispatcher(__name__)

#: Largest ``similarity_top_k`` the legislation endpoint accepts.
MAX_TOP_K = 50

T = TypeVar("T")

# Anything shaped like a LexAPI key is masked in error text, as a last
# line of defence: no message this package builds contains the key.
_KEY_PATTERN = re.compile(r"lex_[A-Za-z0-9_\-]{6,}")


class LexAPIToolError(Exception):
    """A LexAPI call made by a tool failed.

    The message names the API's error code, the HTTP status and the API's
    message, for example ``LexAPI error CREDITS_EXHAUSTED (HTTP 402):
    Monthly credit pool exhausted``. LlamaIndex agents pass it to the model
    as the tool's error output. The SDK's exception is kept as
    ``__cause__``.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status: int | None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.retry_after = retry_after

    @classmethod
    def from_sdk(cls, error: LexAPIError) -> "LexAPIToolError":
        status = f" (HTTP {error.status})" if error.status is not None else ""
        message = f"LexAPI error {error.code}{status}: {error.message}"
        if error.retry_after is not None:
            message += f" Retry after {error.retry_after:g} s."
        return cls(
            _KEY_PATTERN.sub("lex_***", message),
            code=error.code,
            status=error.status,
            retry_after=error.retry_after,
        )


class LexAPIToolEvent(BaseEvent):
    """Dispatched once per successful tool call, with the
    :class:`~lexapi.passages.PassageResult` of that call (credits charged
    and left, withheld/dropped counts)."""

    tool: str
    result: PassageResult

    @classmethod
    def class_name(cls) -> str:
        return "LexAPIToolEvent"


def _call(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except LexAPIError as error:
        raise LexAPIToolError.from_sdk(error) from error


async def _acall(coro: Any) -> Any:
    try:
        return await coro
    except LexAPIError as error:
        raise LexAPIToolError.from_sdk(error) from error


def _render(result: PassageResult, empty: str) -> str:
    blocks = [passages.format_passage(p) for p in result.passages]
    if not blocks:
        blocks = [empty]
        if result.dropped:
            blocks.append(
                f"{result.dropped} hit(s) left out: text for {result.as_of} not established."
            )
    if result.hint:
        blocks.append(f"Hint from LexAPI: {result.hint}")
    return "\n\n".join(blocks)


class LexAPIToolSpec(BaseToolSpec):
    """LexAPI tools for EU legislation.

    - ``lexapi_search_legislation``: semantic search over EU legislation,
      optionally for a date.
    - ``lexapi_read_document``: a document by CELEX number, or one article.
    - ``lexapi_law_on_date``: the version of an act LexAPI lists for a date,
      and optionally one provision of it.

    Each tool has an async variant (``a`` + name) used by async agents.
    Errors raise :class:`LexAPIToolError` with the API's code and message.

    Args:
        client: a :class:`lexapi.LexAPI` to use. Not closed by the tool spec.
        async_client: a :class:`lexapi.AsyncLexAPI` for the async variants.
            With only ``client``, async calls run it in a worker thread.
        api_key: build the clients from this key instead (default: the
            ``LEXAPI_API_KEY`` environment variable). Not stored here.
        base_url: base URL for clients built from a key.
        similarity_top_k: hits requested per search (1-50).
        language: default language of texts (ISO 639-1).
        min_score: similarity floor for searches.
        unresolved: policy for hits whose text for the date asked was not
            established: ``"withhold"`` (default), ``"drop"`` or
            ``"keep_labelled"`` (see :mod:`lexapi.passages`).
    """

    spec_functions: list[SPEC_FUNCTION_TYPE] = [  # noqa: RUF012 - LlamaIndex convention
        ("lexapi_search_legislation", "alexapi_search_legislation"),
        ("lexapi_read_document", "alexapi_read_document"),
        ("lexapi_law_on_date", "alexapi_law_on_date"),
    ]

    def __init__(
        self,
        client: LexAPI | None = None,
        async_client: AsyncLexAPI | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        similarity_top_k: int = 5,
        language: str | None = None,
        min_score: float | None = None,
        unresolved: UnresolvedPolicy = "withhold",
    ) -> None:
        if (
            isinstance(similarity_top_k, bool)
            or not isinstance(similarity_top_k, int)
            or not 1 <= similarity_top_k <= MAX_TOP_K
        ):
            raise ValueError(
                f"similarity_top_k must be an integer from 1 to {MAX_TOP_K}; "
                f"got {similarity_top_k!r}"
            )
        if unresolved not in get_args(UnresolvedPolicy):
            raise ValueError(
                f"unresolved must be one of {', '.join(get_args(UnresolvedPolicy))}; "
                f"got {unresolved!r}"
            )
        self._similarity_top_k = similarity_top_k
        self._language = language
        self._min_score = min_score
        self._unresolved: UnresolvedPolicy = unresolved
        self._clients = Clients(client, async_client, api_key, base_url)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(similarity_top_k={self._similarity_top_k}, "
            f"unresolved={self._unresolved!r})"
        )

    def close(self) -> None:
        """Close the sync client built from a key; a client passed in is left open."""
        self._clients.close()

    async def aclose(self) -> None:
        """Close the async client built from a key."""
        await self._clients.aclose()

    def to_tool_list(
        self,
        spec_functions: list[SPEC_FUNCTION_TYPE] | None = None,
        func_to_metadata_mapping: dict[str, ToolMetadata] | None = None,
    ) -> list[FunctionTool]:
        """The tools, with fixed names, descriptions and argument schemas
        (the same on every supported ``llama-index-core`` version).
        ``func_to_metadata_mapping`` entries override them."""
        mapping = {**tool_metadata(), **(func_to_metadata_mapping or {})}
        return super().to_tool_list(spec_functions, mapping)

    # ── helpers ─────────────────────────────────────────────────────────

    def _emit(self, tool: str, result: PassageResult) -> None:
        dispatcher.event(LexAPIToolEvent(tool=tool, result=result))

    async def _in_thread(self, fn: Callable[..., str], *args: Any) -> str:
        return await asyncio.to_thread(fn, *args)

    def _search_kwargs(self, as_of: str | None) -> dict[str, Any]:
        return {
            "limit": self._similarity_top_k,
            "include_text": True,
            "language": self._language,
            "min_score": self._min_score,
            "as_of": as_of,
        }

    def _search_result(self, response: SemanticLegislationResponse, as_of: str | None) -> str:
        result = passages.from_legislation_search(
            response, unresolved=self._unresolved, as_of=as_of
        )
        self._emit("lexapi_search_legislation", result)
        return _render(result, "LexAPI returned no matching articles.")

    def _document_kwargs(self, article: str | None, language: str | None) -> dict[str, Any]:
        return {
            "language": language or self._language,
            "include": ["metadata", "articles"] if article else None,
            "article_id": article,
        }

    def _document_result(self, response: DocumentResponse, celex: str, article: str | None) -> str:
        result = passages.from_document(response)
        self._emit("lexapi_read_document", result)
        part = f" {article}" if article else ""
        return _render(result, f"LexAPI returned no text for CELEX {celex}{part}.")

    def _as_of_result(
        self, response: LegislationAsOfResponse, celex: str, provision: str | None
    ) -> str:
        result = passages.from_legislation_as_of(
            response, provision=provision, unresolved=self._unresolved
        )
        self._emit("lexapi_law_on_date", result)
        return _render(result, f"LexAPI returned no answer for CELEX {celex}.")

    # ── tools ───────────────────────────────────────────────────────────

    def lexapi_search_legislation(self, query: str, as_of: str | None = None) -> str:
        """Search EU legislation by meaning; returns matching articles, each with its citation.

        Args:
            query: What to look for, in plain language.
            as_of: Optional date, YYYY-MM-DD. Each article is then given in the version LexAPI
                lists for that date; when that could not be established the article is returned
                as a notice without text, with the reason.
        """
        response = _call(
            lambda: self._clients.sync.semantic_legislation_search(
                query, **self._search_kwargs(as_of)
            )
        )
        return self._search_result(response, as_of)

    async def alexapi_search_legislation(self, query: str, as_of: str | None = None) -> str:
        """Async variant of lexapi_search_legislation."""
        client = self._clients.async_
        if client is None:
            return await self._in_thread(self.lexapi_search_legislation, query, as_of)
        response = await _acall(
            client.semantic_legislation_search(query, **self._search_kwargs(as_of))
        )
        return self._search_result(response, as_of)

    def lexapi_read_document(
        self, celex: str, article: str | None = None, language: str | None = None
    ) -> str:
        """Read an EU document by its CELEX number: one article, or every article.

        Args:
            celex: CELEX number of the document, e.g. 32016R0679.
            article: Optional article to read, e.g. art_17 or 17. Without it every article of
                the document is returned, which can be long.
            language: Optional language code of the text, e.g. en, de, fr.
        """
        response = _call(
            lambda: self._clients.sync.get_document(
                celex, **self._document_kwargs(article, language)
            )
        )
        return self._document_result(response, celex, article)

    async def alexapi_read_document(
        self, celex: str, article: str | None = None, language: str | None = None
    ) -> str:
        """Async variant of lexapi_read_document."""
        client = self._clients.async_
        if client is None:
            return await self._in_thread(self.lexapi_read_document, celex, article, language)
        response = await _acall(
            client.get_document(celex, **self._document_kwargs(article, language))
        )
        return self._document_result(response, celex, article)

    def lexapi_law_on_date(
        self,
        celex: str,
        date: str,
        provision: str | None = None,
        language: str | None = None,
    ) -> str:
        """The version of an EU act that LexAPI lists for a date, and optionally one provision.

        Args:
            celex: CELEX number of the act, e.g. 32016R0679.
            date: The date, YYYY-MM-DD.
            provision: Optional provision, e.g. art_17, anx_III or rct_26. Without it only the
                version is named.
            language: Optional language code of the text, e.g. en, de, fr.
        """
        response = _call(
            lambda: self._clients.sync.legislation_as_of(
                celex, date, provision=provision, language=language or self._language
            )
        )
        return self._as_of_result(response, celex, provision)

    async def alexapi_law_on_date(
        self,
        celex: str,
        date: str,
        provision: str | None = None,
        language: str | None = None,
    ) -> str:
        """Async variant of lexapi_law_on_date."""
        client = self._clients.async_
        if client is None:
            return await self._in_thread(self.lexapi_law_on_date, celex, date, provision, language)
        response = await _acall(
            client.legislation_as_of(
                celex, date, provision=provision, language=language or self._language
            )
        )
        return self._as_of_result(response, celex, provision)


# ── tool metadata ───────────────────────────────────────────────────────


class SearchLegislationArgs(BaseModel):
    query: str = Field(description="What to look for, in plain language.")
    as_of: str | None = Field(
        default=None,
        description=(
            "Optional date, YYYY-MM-DD. Each article is then given in the version LexAPI lists "
            "for that date; when that could not be established the article is returned as a "
            "notice without text, with the reason."
        ),
    )


class ReadDocumentArgs(BaseModel):
    celex: str = Field(description="CELEX number of the document, e.g. 32016R0679.")
    article: str | None = Field(
        default=None,
        description=(
            "Optional article to read, e.g. art_17 or 17. Without it every article of the "
            "document is returned, which can be long."
        ),
    )
    language: str | None = Field(
        default=None, description="Optional language code of the text, e.g. en, de, fr."
    )


class LawOnDateArgs(BaseModel):
    celex: str = Field(description="CELEX number of the act, e.g. 32016R0679.")
    date: str = Field(description="The date, YYYY-MM-DD.")
    provision: str | None = Field(
        default=None,
        description=(
            "Optional provision, e.g. art_17, anx_III or rct_26. Without it only the version "
            "is named."
        ),
    )
    language: str | None = Field(
        default=None, description="Optional language code of the text, e.g. en, de, fr."
    )


_SCHEMAS: dict[str, type[BaseModel]] = {
    "lexapi_search_legislation": SearchLegislationArgs,
    "lexapi_read_document": ReadDocumentArgs,
    "lexapi_law_on_date": LawOnDateArgs,
}


def _description(name: str) -> str:
    fn = getattr(LexAPIToolSpec, name)
    signature = inspect.signature(fn)
    signature = signature.replace(parameters=list(signature.parameters.values())[1:])
    summary = inspect.cleandoc(fn.__doc__ or "").split("\n\n", 1)[0]
    return f"{name}{signature}\n{summary}"


def tool_metadata() -> dict[str, ToolMetadata]:
    """Name, description and argument schema of each tool."""
    return {
        name: ToolMetadata(name=name, description=_description(name), fn_schema=schema)
        for name, schema in _SCHEMAS.items()
    }
