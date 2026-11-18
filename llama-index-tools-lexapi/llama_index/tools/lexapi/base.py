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

from lexapi import (
    AsyncLexAPI,
    CreditsExhaustedError,
    LexAPI,
    LexAPIError,
    PaymentPastDueError,
    passages,
)
from lexapi.models import DocumentResponse, LegislationAsOfResponse, SemanticLegislationResponse
from lexapi.passages import PassageResult, UnresolvedPolicy
from llama_index.tools.lexapi._client import Clients
from llama_index.tools.lexapi._passages import (
    MIN_MAX_CHARS,
    LexAPICallReport,
    dedupe,
    truncate,
)

dispatcher = instrument.get_dispatcher(__name__)

#: Largest ``similarity_top_k`` the legislation endpoint accepts.
MAX_TOP_K = 50
#: Default character budget of ``lexapi_read_document``.
DEFAULT_READ_MAX_CHARS = 20_000

T = TypeVar("T")

# Anything shaped like a LexAPI key is masked in error text, as a last
# line of defence: no message this package builds contains the key.
_KEY_PATTERN = re.compile(r"lex_[A-Za-z0-9_-]+")


def _mask(text: str) -> str:
    return _KEY_PATTERN.sub("lex_***", text)


class LexAPIToolError(Exception):
    """A LexAPI call made by a tool failed.

    The message is ``LexAPI error <code> (HTTP <status>): <message>``,
    followed when known by ``Credits reset at <time>.``, ``Billing: <url>``
    and ``Retry after <n> s.``. LlamaIndex agents pass it to the model as
    the tool's error output. The SDK's exception is kept as ``__cause__``.
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
        api_message = error.message.strip()
        if api_message and api_message[-1] not in ".!?":
            api_message += "."
        message = f"LexAPI error {error.code}{status}: {api_message}"
        if isinstance(error, CreditsExhaustedError) and error.resets_at:
            message += f" Credits reset at {error.resets_at}."
        if isinstance(error, PaymentPastDueError) and error.billing_url:
            message += f" Billing: {error.billing_url}"
        if error.retry_after is not None:
            message += f" Retry after {error.retry_after:g} s."
        return cls(
            _mask(message), code=error.code, status=error.status, retry_after=error.retry_after
        )


class LexAPIToolEvent(BaseEvent):
    """Dispatched once per successful tool call, with that call's
    :class:`LexAPICallReport` and :class:`~lexapi.passages.PassageResult`."""

    tool: str
    report: LexAPICallReport
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


def _render(
    result: PassageResult,
    *,
    requested: int,
    empty: str,
    max_chars: int | None = None,
    celex: str | None = None,
) -> tuple[str, LexAPICallReport]:
    """The tool output for one result, and the call's report.

    With ``max_chars``, the first passage's text is cut to fit and further
    passages are shown while the output stays within the budget; the rest
    are listed by reference at the end.
    """
    kept, _, duplicates = dedupe(result.passages)
    blocks: list[str] = []
    shown = 0
    truncated = 0
    total = 0
    for passage in kept:
        candidate = passage
        if max_chars is not None:
            candidate, cut = truncate(passage, max_chars)
            block = passages.format_passage(candidate)
            if blocks and total + 2 + len(block) > max_chars:
                break
            truncated += cut
        else:
            block = passages.format_passage(candidate)
        blocks.append(block)
        total += len(block) + (2 if shown else 0)
        shown += 1
    rest = kept[shown:]
    if rest:
        refs = ", ".join(p.article_ref or "?" for p in rest)
        blocks.append(
            f"[Not shown: {len(rest)} more article(s) of CELEX {celex}: {refs}. "
            "Read one with the article argument.]"
        )
    if not kept:
        blocks = [empty]
        if result.dropped:
            blocks.append(
                f"{result.dropped} hit(s) left out: text for {result.as_of} not established."
            )
    if result.hint:
        blocks.append(f"Hint from LexAPI: {result.hint}")
    report = LexAPICallReport(
        requested=requested,
        returned=shown,
        as_of=result.as_of,
        withheld=result.withheld,
        kept_labelled=result.kept_labelled,
        dropped=result.dropped,
        uncitable=result.uncitable,
        filtered_out=result.filtered_out,
        withheld_not_emitted=0,
        returned_withheld=sum(p.text_withheld for p in kept[:shown]),
        duplicates=duplicates,
        truncated=truncated,
        units_charged=result.units_charged,
        credits_remaining=result.credits_remaining,
        hint=result.hint,
    )
    return "\n\n".join(blocks), report


class LexAPIToolSpec(BaseToolSpec):
    """LexAPI tools for EU legislation.

    - ``lexapi_search_legislation``: semantic search over EU legislation,
      optionally for a date.
    - ``lexapi_read_document``: a document by CELEX number, or one article.
    - ``lexapi_law_on_date``: the version of an act LexAPI lists for a date,
      and optionally one provision of it. An answer LexAPI could not
      establish is always a notice without text.

    Each tool has an async variant (``a`` + name) used by async agents.
    Errors raise :class:`LexAPIToolError` with the API's code and message.

    Args:
        client: a :class:`lexapi.LexAPI` to use. Never closed by the tool spec.
        async_client: a :class:`lexapi.AsyncLexAPI` for the async variants.
            With only ``client``, async calls run it in a worker thread.
        api_key: build the clients from this key instead (default: the
            ``LEXAPI_API_KEY`` environment variable). The sync client is
            built at construction, the async one on first async use; close
            them with :meth:`close` / :meth:`aclose`.
        base_url: base URL for clients built from a key.
        similarity_top_k: hits requested per search (1-50).
        language: default language of texts (ISO 639-1).
        min_score: similarity floor for searches.
        unresolved: for searches with a date, what to do with an article
            whose version for that date LexAPI could not establish:
            ``"withhold"`` (default; a notice without text), ``"drop"`` (left
            out) or ``"keep_labelled"`` (the latest indexed text, labelled).
            The tool descriptions follow this setting.
        max_chars: character budget of ``lexapi_read_document`` output (at
            least 200, default 20000). Articles beyond it are listed by
            reference.
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
        max_chars: int = DEFAULT_READ_MAX_CHARS,
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
        if (
            isinstance(max_chars, bool)
            or not isinstance(max_chars, int)
            or max_chars < MIN_MAX_CHARS
        ):
            raise ValueError(
                f"max_chars must be an integer of at least {MIN_MAX_CHARS}; got {max_chars!r}"
            )
        self._similarity_top_k = similarity_top_k
        self._language = language
        self._min_score = min_score
        self._unresolved: UnresolvedPolicy = unresolved
        self._max_chars = max_chars
        self._clients = Clients(client, async_client, api_key, base_url)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(similarity_top_k={self._similarity_top_k}, "
            f"unresolved={self._unresolved!r}, max_chars={self._max_chars})"
        )

    def close(self) -> None:
        """Close the sync client built from a key; a client passed in is left open."""
        self._clients.close()

    async def aclose(self) -> None:
        """Close the async client built from a key, if it was built."""
        await self._clients.aclose()

    def to_tool_list(
        self,
        spec_functions: list[SPEC_FUNCTION_TYPE] | None = None,
        func_to_metadata_mapping: dict[str, ToolMetadata] | None = None,
    ) -> list[FunctionTool]:
        """The tools, with fixed names, descriptions and argument schemas
        (the same on every supported ``llama-index-core`` version; the
        search tool's text follows ``unresolved``).
        ``func_to_metadata_mapping`` entries override them."""
        mapping = {**tool_metadata(self._unresolved), **(func_to_metadata_mapping or {})}
        return super().to_tool_list(spec_functions, mapping)

    # ── helpers ─────────────────────────────────────────────────────────

    def _emit(self, tool: str, report: LexAPICallReport, result: PassageResult) -> None:
        dispatcher.event(LexAPIToolEvent(tool=tool, report=report, result=result))

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
        text, report = _render(
            result,
            requested=self._similarity_top_k,
            empty="LexAPI returned no matching articles.",
        )
        self._emit("lexapi_search_legislation", report, result)
        return text

    def _document_kwargs(self, article: str | None, language: str | None) -> dict[str, Any]:
        return {
            "language": language or self._language,
            "include": ["metadata", "articles"] if article else None,
            "article_id": article,
        }

    def _document_result(self, response: DocumentResponse, celex: str, article: str | None) -> str:
        result = passages.from_document(response)
        part = f" {article}" if article else ""
        text, report = _render(
            result,
            requested=1,
            empty=f"LexAPI returned no text for CELEX {celex}{part}.",
            max_chars=self._max_chars,
            celex=celex,
        )
        self._emit("lexapi_read_document", report, result)
        return text

    def _as_of_result(
        self, response: LegislationAsOfResponse, celex: str, provision: str | None
    ) -> str:
        # Always withheld: an answer LexAPI could not establish carries no
        # text, whatever the spec's search policy.
        result = passages.from_legislation_as_of(
            response, provision=provision, unresolved="withhold"
        )
        text, report = _render(
            result, requested=1, empty=f"LexAPI returned no answer for CELEX {celex}."
        )
        self._emit("lexapi_law_on_date", report, result)
        return text

    # ── tools ───────────────────────────────────────────────────────────

    def lexapi_search_legislation(self, query: str, as_of: str | None = None) -> str:
        """Search EU legislation by meaning; returns matching articles, each with its citation.

        Args:
            query: What to look for, in plain language.
            as_of: Optional date, YYYY-MM-DD. What happens to an article whose version for that
                date could not be established follows the spec's ``unresolved`` setting.
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
            article: Optional article to read, e.g. art_17 or 17. Without it the articles of
                the document are returned up to a size limit; the rest are listed.
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

_AS_OF_LEAD = (
    "Optional date, YYYY-MM-DD. Each article is then given in the version LexAPI lists for "
    "that date; "
)
#: What the search tool tells the model about an article whose version for
#: the date asked could not be established, per ``unresolved`` policy.
AS_OF_POLICY_SENTENCES: dict[str, str] = {
    "withhold": (
        "an article whose version for that date could not be established is returned as a "
        "notice without text, with the reason."
    ),
    "drop": "articles whose version for that date could not be established are left out.",
    "keep_labelled": (
        "an article whose version for that date could not be established is returned with "
        "its latest indexed text, labelled as not established for that date."
    ),
}


def _search_schema(policy: str) -> type[BaseModel]:
    as_of_description = _AS_OF_LEAD + AS_OF_POLICY_SENTENCES[policy]

    class SearchLegislationArgs(BaseModel):
        query: str = Field(description="What to look for, in plain language.")
        as_of: str | None = Field(default=None, description=as_of_description)

    return SearchLegislationArgs


class ReadDocumentArgs(BaseModel):
    celex: str = Field(description="CELEX number of the document, e.g. 32016R0679.")
    article: str | None = Field(
        default=None,
        description=(
            "Optional article to read, e.g. art_17 or 17. Without it the articles of the "
            "document are returned up to a size limit; the rest are listed by reference."
        ),
    )
    language: str | None = Field(
        default=None, description="Optional language code of the text, e.g. en, de, fr."
    )


class LawOnDateArgs(BaseModel):
    celex: str = Field(description="CELEX number of the act, e.g. 32016R0679.")
    date: str = Field(description=("The date, YYYY-MM-DD."))
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


_LAW_ON_DATE_NOTE = (
    "When LexAPI could not establish the version for that date, the answer is a notice "
    "without text, with the reason."
)


def _description(name: str, extra: str = "") -> str:
    fn = getattr(LexAPIToolSpec, name)
    signature = inspect.signature(fn)
    signature = signature.replace(parameters=list(signature.parameters.values())[1:])
    summary = inspect.cleandoc(fn.__doc__ or "").split("\n\n", 1)[0]
    return f"{name}{signature}\n{summary}" + (f" {extra}" if extra else "")


def tool_metadata(unresolved: UnresolvedPolicy = "withhold") -> dict[str, ToolMetadata]:
    """Name, description and argument schema of each tool, for the given
    search policy."""
    schemas: dict[str, type[BaseModel]] = {
        "lexapi_search_legislation": _search_schema(unresolved),
        "lexapi_read_document": ReadDocumentArgs,
        "lexapi_law_on_date": LawOnDateArgs,
    }
    extras = {
        "lexapi_search_legislation": "With as_of, " + AS_OF_POLICY_SENTENCES[unresolved],
        "lexapi_law_on_date": _LAW_ON_DATE_NOTE,
    }
    return {
        name: ToolMetadata(
            name=name, description=_description(name, extras.get(name, "")), fn_schema=schema
        )
        for name, schema in schemas.items()
    }
