"""LexAPI retrievers: EU legislation and CJEU case law by meaning.

Both retrievers convert responses with :mod:`lexapi.passages` only; see
:mod:`llama_index.retrievers.lexapi._nodes` for the node mapping.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import re
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple, NoReturn, Self, SupportsIndex, get_args

import llama_index.core.instrumentation as instrument
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType

from lexapi import AsyncLexAPI, LexAPI, passages
from lexapi.models import SemanticLegislationResponse, SemanticSearchResponse
from lexapi.passages import PassageResult, UnresolvedPolicy
from llama_index.retrievers.lexapi._client import Clients
from llama_index.retrievers.lexapi._nodes import MIN_MAX_CHARS, LexAPICallReport, build

dispatcher = instrument.get_dispatcher(__name__)

#: Largest ``similarity_top_k`` the legislation endpoint accepts. A plan can
#: allow fewer (the API clamps the limit to the plan).
MAX_LEGISLATION_TOP_K = 50
#: Largest ``similarity_top_k`` the case-law endpoint accepts on any plan.
MAX_CASE_LAW_TOP_K = 100

DateLike = dt.date | str


class LexAPIRetrieval(NamedTuple):
    """What one retrieval returned: the nodes, the call's
    :class:`LexAPICallReport`, and the
    :class:`lexapi.passages.PassageResult` the nodes were built from."""

    nodes: list[NodeWithScore]
    report: LexAPICallReport
    result: PassageResult


class LexAPIRetrievalEvent(BaseEvent):
    """Dispatched once per retrieval, with that call's
    :class:`LexAPICallReport` and :class:`~lexapi.passages.PassageResult`.

    Subscribe with an instrumentation event handler to read the credits a
    retrieval cost when the retriever runs inside a query engine. The event
    belongs to the retrieval's span (``span_id``).
    """

    retriever: str
    query: str
    report: LexAPICallReport
    result: PassageResult

    @classmethod
    def class_name(cls) -> str:
        return "LexAPIRetrievalEvent"


# Per-call holder for retrieve_with_result(): a fresh dict for each call,
# read back by that call only. Context variables are per thread and per
# asyncio task, so concurrent calls never see each other's result.
_Captured = tuple[LexAPICallReport, PassageResult]
_RESULT_HOLDER: contextvars.ContextVar[dict[int, _Captured] | None] = contextvars.ContextVar(
    "lexapi_retrieval_result", default=None
)


def _check_top_k(value: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"similarity_top_k must be an integer from 1 to {maximum}; got {value!r}")
    return value


def _check_max_chars(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < MIN_MAX_CHARS:
        raise ValueError(
            f"max_chars must be an integer of at least {MIN_MAX_CHARS}, or None; got {value!r}"
        )
    return value


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _iso(value: DateLike | None, name: str = "as_of") -> str | None:
    """``YYYY-MM-DD`` for a date, or a string that is a real calendar day in
    that form; anything else raises ``ValueError`` before any request."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, str) and _ISO_DATE.fullmatch(value):
        try:
            dt.date.fromisoformat(value)
        except ValueError:
            pass
        else:
            return value
    raise ValueError(f"{name} must be a date in the form YYYY-MM-DD; got {value!r}")


class _LexAPIRetriever(BaseRetriever):
    """Shared plumbing: client use, result capture, events."""

    def __init__(
        self,
        clients: Clients,
        *,
        callback_manager: CallbackManager | None,
        verbose: bool,
    ) -> None:
        self._clients = clients
        super().__init__(callback_manager=callback_manager, verbose=verbose)

    # ── subclass hooks ──────────────────────────────────────────────────

    def _search(self, query: str) -> PassageResult:
        raise NotImplementedError

    async def _asearch(self, client: AsyncLexAPI, query: str) -> PassageResult:
        raise NotImplementedError

    def _build(self, result: PassageResult) -> tuple[list[NodeWithScore], LexAPICallReport]:
        raise NotImplementedError

    # ── BaseRetriever ───────────────────────────────────────────────────

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # No try/except: an SDK error (402, 403 past due, 429, 401...)
        # propagates. It must never read as "no law found".
        result = self._search(query_bundle.query_str)
        return self._finish(query_bundle, result)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        client = self._clients.async_
        if client is None:
            result = await asyncio.to_thread(self._search, query_bundle.query_str)
        else:
            result = await self._asearch(client, query_bundle.query_str)
        return self._finish(query_bundle, result)

    def _finish(self, query_bundle: QueryBundle, result: PassageResult) -> list[NodeWithScore]:
        nodes, report = self._build(result)
        holder = _RESULT_HOLDER.get()
        if holder is not None:
            holder.setdefault(id(self), (report, result))
        dispatcher.event(
            LexAPIRetrievalEvent(
                retriever=type(self).__name__,
                query=query_bundle.query_str,
                report=report,
                result=result,
            )
        )
        return nodes

    # ── lifecycle ───────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _no_copy(self) -> NoReturn:
        raise TypeError(
            f"{type(self).__name__} holds an open client and cannot be copied or pickled."
        )

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        self._no_copy()

    def __copy__(self) -> NoReturn:
        self._no_copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        self._no_copy()

    # ── public extras ───────────────────────────────────────────────────

    def retrieve_with_result(self, str_or_query_bundle: QueryType) -> LexAPIRetrieval:
        """:meth:`retrieve`, also returning the call's
        :class:`LexAPICallReport` (hits requested and nodes returned, credits
        charged and left, withheld/dropped/filtered/duplicate counts, hint)
        and :class:`~lexapi.passages.PassageResult`.

        Safe under concurrency: the result is carried per call, not stored
        on the retriever.
        """
        holder: dict[int, _Captured] = {}
        token = _RESULT_HOLDER.set(holder)
        try:
            nodes = self.retrieve(str_or_query_bundle)
        finally:
            _RESULT_HOLDER.reset(token)
        report, result = holder[id(self)]
        return LexAPIRetrieval(nodes, report, result)

    async def aretrieve_with_result(self, str_or_query_bundle: QueryType) -> LexAPIRetrieval:
        """Async :meth:`retrieve_with_result`."""
        holder: dict[int, _Captured] = {}
        token = _RESULT_HOLDER.set(holder)
        try:
            nodes = await self.aretrieve(str_or_query_bundle)
        finally:
            _RESULT_HOLDER.reset(token)
        report, result = holder[id(self)]
        return LexAPIRetrieval(nodes, report, result)

    def close(self) -> None:
        """Close the sync client this retriever built from a key and mark
        the retriever closed. A client passed in is never closed."""
        self._clients.close()

    async def aclose(self) -> None:
        """Close both clients this retriever built from a key and mark the
        retriever closed. A client passed in is never closed."""
        await self._clients.aclose()


class LexAPILegislationRetriever(_LexAPIRetriever):
    """Retrieve articles of EU legislation from LexAPI's semantic search,
    each with the citation it must be shown with.

    Each hit becomes one node: the text is the article followed by a
    ``Source:`` line (:func:`lexapi.passages.format_passage`), the metadata
    is :meth:`lexapi.passages.Passage.metadata`, the id is
    :attr:`~lexapi.passages.Passage.id` and the score is the API's,
    unchanged. Nodes keep the API's order. Hits sharing an id are merged
    (first position kept, a later hit replaces it only when it carries more
    text), so scores need not be descending: each node keeps its own score.

    Args:
        client: a :class:`lexapi.LexAPI` to use. Not closed by the retriever.
        async_client: a :class:`lexapi.AsyncLexAPI` for :meth:`aretrieve`.
            With only ``client``, async calls run it in a worker thread.
        api_key: build the clients from this key instead (default: the
            ``LEXAPI_API_KEY`` environment variable). The sync client is
            built at construction, the async client on first async use; close
            them with :meth:`close` / :meth:`aclose`.
        base_url: base URL for clients built from a key (default: the SDK's).
        similarity_top_k: number of hits requested from the API (1-50; the
            plan can allow fewer). The result can hold fewer nodes, and
            never more.
        as_of: a date (``YYYY-MM-DD`` or :class:`datetime.date`). Each hit
            then carries the version LexAPI lists for that date, when it
            could establish it. The same date is checked on every hit.
        language: language of the texts (ISO 639-1; default: the API's).
        min_score: similarity floor sent to the API.
        celex: keep only hits of this act or these acts (CELEX numbers;
            consolidated texts of an act are kept with it). The search cannot
            be limited to an act, so this filters the ``similarity_top_k``
            hits returned: the result can be shorter or empty.
        unresolved: what to do with a hit whose text for ``as_of`` LexAPI did
            not establish: ``"withhold"`` (default) keeps it as a node whose
            text is a notice with the reason and no text; ``"drop"`` leaves it
            out; ``"keep_labelled"`` keeps the latest indexed text, labelled
            as not established for the date.
        emit_withheld: ``False`` leaves withheld hits out of the nodes
            (counted in ``report.withheld_not_emitted``). Withheld nodes carry
            ``metadata["text_withheld"] = True``; do not index or embed them.
        include_citation_in_text: ``True`` (default) puts the ``Source:``
            line in the node text, so the citation reaches the model in
            every response mode. ``False`` uses the text alone.
        callback_manager, verbose: passed to ``BaseRetriever``.
    """

    def __init__(
        self,
        client: LexAPI | None = None,
        async_client: AsyncLexAPI | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        similarity_top_k: int = 5,
        as_of: DateLike | None = None,
        language: str | None = None,
        min_score: float | None = None,
        celex: str | Sequence[str] | None = None,
        unresolved: UnresolvedPolicy = "withhold",
        emit_withheld: bool = True,
        include_citation_in_text: bool = True,
        callback_manager: CallbackManager | None = None,
        verbose: bool = False,
    ) -> None:
        self._similarity_top_k = _check_top_k(similarity_top_k, MAX_LEGISLATION_TOP_K)
        if unresolved not in get_args(UnresolvedPolicy):
            raise ValueError(
                f"unresolved must be one of {', '.join(get_args(UnresolvedPolicy))}; "
                f"got {unresolved!r}"
            )
        self._unresolved: UnresolvedPolicy = unresolved
        self._as_of = _iso(as_of)
        self._language = language
        self._min_score = min_score
        self._celex = _celex_tuple(celex)
        self._emit_withheld = emit_withheld
        self._include_citation_in_text = include_citation_in_text
        super().__init__(
            Clients(client, async_client, api_key, base_url, type(self).__name__),
            callback_manager=callback_manager,
            verbose=verbose,
        )

    @property
    def similarity_top_k(self) -> int:
        return self._similarity_top_k

    @property
    def as_of(self) -> str | None:
        return self._as_of

    @property
    def unresolved(self) -> UnresolvedPolicy:
        return self._unresolved

    @property
    def celex(self) -> tuple[str, ...] | None:
        return self._celex

    def _request(self) -> dict[str, Any]:
        return {
            "limit": self._similarity_top_k,
            "include_text": True,
            "language": self._language,
            "min_score": self._min_score,
            "as_of": self._as_of,
        }

    def _search(self, query: str) -> PassageResult:
        response = self._clients.sync.semantic_legislation_search(query, **self._request())
        return self._convert(response)

    async def _asearch(self, client: AsyncLexAPI, query: str) -> PassageResult:
        response = await client.semantic_legislation_search(query, **self._request())
        return self._convert(response)

    def _convert(self, response: SemanticLegislationResponse) -> PassageResult:
        result = passages.from_legislation_search(
            response, unresolved=self._unresolved, as_of=self._as_of
        )
        if self._celex is not None:
            result = passages.restrict_to_celex(result, self._celex)
        return result

    def _build(self, result: PassageResult) -> tuple[list[NodeWithScore], LexAPICallReport]:
        return build(
            result,
            requested=self._similarity_top_k,
            include_citation_in_text=self._include_citation_in_text,
            emit_withheld=self._emit_withheld,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(similarity_top_k={self._similarity_top_k}, "
            f"as_of={self._as_of!r}, unresolved={self._unresolved!r}, celex={self._celex!r})"
        )


class LexAPICaseLawRetriever(_LexAPIRetriever):
    """Retrieve CJEU case law from LexAPI's semantic search, each decision
    with the citation it must be shown with.

    The search is always sent with ``include_text=True``. The text of a node
    is the whole decision; decisions are long. ``max_chars`` cuts it at a
    paragraph break when one is near, otherwise at a line break, a space or
    the limit, and marks the cut. The snippet and the machine-written
    summary are never used as text: a hit the API sent without text becomes
    a node holding only a notice that no text was returned.

    Args:
        client, async_client, api_key, base_url: as for
            :class:`LexAPILegislationRetriever`.
        similarity_top_k: number of hits requested from the API (1-100; the
            plan can allow fewer). The result can hold fewer nodes.
        language: language of the texts (ISO 639-1).
        min_score: similarity floor sent to the API.
        court: keep to one court (sent as a search filter).
        date_from, date_to: decision-date range (sent as search filters).
        max_chars: cut each decision's text to at most this many characters
            (at least 200; default ``None``: the whole decision).
        include_citation_in_text: as for :class:`LexAPILegislationRetriever`.
        callback_manager, verbose: passed to ``BaseRetriever``.
    """

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
        court: str | None = None,
        date_from: DateLike | None = None,
        date_to: DateLike | None = None,
        max_chars: int | None = None,
        include_citation_in_text: bool = True,
        callback_manager: CallbackManager | None = None,
        verbose: bool = False,
    ) -> None:
        self._similarity_top_k = _check_top_k(similarity_top_k, MAX_CASE_LAW_TOP_K)
        self._language = language
        self._min_score = min_score
        filters = {
            "court": court,
            "date_from": _iso(date_from, "date_from"),
            "date_to": _iso(date_to, "date_to"),
        }
        self._filters = {k: v for k, v in filters.items() if v is not None} or None
        self._max_chars = _check_max_chars(max_chars)
        self._include_citation_in_text = include_citation_in_text
        super().__init__(
            Clients(client, async_client, api_key, base_url, type(self).__name__),
            callback_manager=callback_manager,
            verbose=verbose,
        )

    @property
    def similarity_top_k(self) -> int:
        return self._similarity_top_k

    @property
    def max_chars(self) -> int | None:
        return self._max_chars

    def _request(self) -> dict[str, Any]:
        return {
            "limit": self._similarity_top_k,
            "include_text": True,
            "language": self._language,
            "min_score": self._min_score,
            "filters": dict(self._filters) if self._filters else None,
        }

    def _search(self, query: str) -> PassageResult:
        response = self._clients.sync.semantic_search(query, **self._request())
        return self._convert(response)

    async def _asearch(self, client: AsyncLexAPI, query: str) -> PassageResult:
        response = await client.semantic_search(query, **self._request())
        return self._convert(response)

    def _convert(self, response: SemanticSearchResponse) -> PassageResult:
        return passages.from_case_law_search(response)

    def _build(self, result: PassageResult) -> tuple[list[NodeWithScore], LexAPICallReport]:
        return build(
            result,
            requested=self._similarity_top_k,
            include_citation_in_text=self._include_citation_in_text,
            max_chars=self._max_chars,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(similarity_top_k={self._similarity_top_k}, "
            f"max_chars={self._max_chars!r})"
        )


def _celex_tuple(celex: str | Iterable[str] | None) -> tuple[str, ...] | None:
    if celex is None:
        return None
    values = (celex,) if isinstance(celex, str) else tuple(celex)
    # The core validates the numbers (strings, at least one).
    passages.restrict_to_celex(PassageResult(passages=()), values)
    return values
