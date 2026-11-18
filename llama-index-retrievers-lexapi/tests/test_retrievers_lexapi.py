"""Behaviour of the LexAPI retrievers, offline, against synthetic fixtures."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import lexapi
import pytest
from lexapi import passages
from lexapi.models import SemanticLegislationResponse, SemanticSearchResponse
from lexapi.passages import PassageResult
from lexapi_offline import (
    LOOPBACK,
    OFFLINE_KEY,
    Route,
    fixture_route,
    load_fixture,
    make_async_client,
    make_client,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode

from llama_index.retrievers.lexapi import (
    EXCLUDED_LLM_METADATA_KEYS,
    LexAPICaseLawRetriever,
    LexAPILegislationRetriever,
    LexAPIRetrievalEvent,
)
from llama_index.retrievers.lexapi._client import user_agent_suffix

LEG = "/legislation/semantic"
CASE = "/search/semantic"
AS_OF = "2099-06-01"

GUARDRAIL_KEYS = (
    "citation",
    "cite_celex",
    "consolidated",
    "as_of",
    "as_of_resolved",
    "as_of_reason",
    "text_status",
)


def leg_retriever(fixture: str, **kwargs: Any) -> tuple[LexAPILegislationRetriever, list[Any]]:
    client, recorder = make_client({LEG: fixture_route(fixture)})
    retriever = LexAPILegislationRetriever(client, **kwargs)
    return retriever, recorder.requests


def leg_expected(fixture: str, **kwargs: Any) -> PassageResult:
    """The core's answer for the same body, to compare the nodes with."""
    response = SemanticLegislationResponse(load_fixture(fixture))
    return passages.from_legislation_search(response, **kwargs)


def text_node(item: NodeWithScore) -> TextNode:
    assert isinstance(item.node, TextNode)
    return item.node


def all_views(item: NodeWithScore) -> list[str]:
    node = text_node(item)
    return [
        node.text,
        json.dumps(node.metadata),
        node.get_content(metadata_mode=MetadataMode.ALL),
        node.get_content(metadata_mode=MetadataMode.LLM),
        node.get_content(metadata_mode=MetadataMode.EMBED),
        node.get_content(metadata_mode=MetadataMode.NONE),
        node.node_id,
        repr(node),
    ]


# ── class shape ─────────────────────────────────────────────────────────


def test_retrievers_derive_from_base_retriever() -> None:
    assert BaseRetriever in LexAPILegislationRetriever.__mro__
    assert BaseRetriever in LexAPICaseLawRetriever.__mro__


# ── request ─────────────────────────────────────────────────────────────


def test_request_body_without_a_date() -> None:
    retriever, requests = leg_retriever("legislation.no-date.json")
    retriever.retrieve("right to erasure")
    (request,) = requests
    assert json.loads(request.content) == {
        "query": "right to erasure",
        "limit": 5,
        "include": ["text"],
    }
    assert request.headers["x-api-key"] == OFFLINE_KEY
    assert request.url.path == "/api/v1" + LEG


def test_request_body_with_a_date_and_options() -> None:
    import datetime as dt

    retriever, requests = leg_retriever(
        "legislation.mixed.json",
        as_of=dt.date(2099, 6, 1),
        similarity_top_k=7,
        language="de",
        min_score=0.4,
    )
    retriever.retrieve("q")
    assert json.loads(requests[0].content) == {
        "query": "q",
        "limit": 7,
        "include": ["text"],
        "language": "de",
        "min_score": 0.4,
        "asOf": AS_OF,
    }


def test_celex_restriction_does_not_over_ask() -> None:
    retriever, requests = leg_retriever(
        "legislation.mixed.json", as_of=AS_OF, celex="32099R0001", similarity_top_k=3
    )
    result = retriever.retrieve_with_result("q")
    assert json.loads(requests[0].content)["limit"] == 3
    assert [text_node(n).metadata["cite_celex"] for n in result.nodes] == ["32099R0001"]
    assert result.result.filtered_out == 2
    assert result.result.withheld == 0


@pytest.mark.parametrize("bad", [0, 51, -1, True, 2.0, "5"])
def test_similarity_top_k_is_validated(bad: Any) -> None:
    client, _ = make_client({})
    with pytest.raises(ValueError, match="similarity_top_k"):
        LexAPILegislationRetriever(client, similarity_top_k=bad)


def test_case_law_top_k_bound() -> None:
    client, _ = make_client({})
    LexAPICaseLawRetriever(client, similarity_top_k=100)
    with pytest.raises(ValueError, match="similarity_top_k"):
        LexAPICaseLawRetriever(client, similarity_top_k=101)


@pytest.mark.parametrize("bad", ["latest_labelled", "keep", ""])
def test_unknown_unresolved_policy_is_refused(bad: str) -> None:
    client, _ = make_client({})
    with pytest.raises(ValueError, match="unresolved"):
        LexAPILegislationRetriever(client, unresolved=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [[], [""], [7]])
def test_celex_is_validated_by_the_core(bad: Any) -> None:
    client, _ = make_client({})
    with pytest.raises(ValueError):
        LexAPILegislationRetriever(client, celex=bad)


# ── node mapping ────────────────────────────────────────────────────────


FIXTURES = [
    ("legislation.as-published.json", AS_OF),
    ("legislation.consolidated.json", AS_OF),
    ("legislation.unresolved.json", AS_OF),
    ("legislation.mixed.json", AS_OF),
    ("legislation.no-date.json", None),
    ("legislation.empty.json", None),
]


@pytest.mark.parametrize(("fixture", "as_of"), FIXTURES)
@pytest.mark.parametrize("policy", ["withhold", "drop", "keep_labelled"])
def test_nodes_are_exactly_the_core_passages(fixture: str, as_of: str | None, policy: Any) -> None:
    retriever, _ = leg_retriever(fixture, as_of=as_of, unresolved=policy)
    nodes = retriever.retrieve("q")
    expected = leg_expected(fixture, unresolved=policy, as_of=as_of).passages
    assert len(nodes) == len(expected)
    for item, passage in zip(nodes, expected, strict=True):
        node = text_node(item)
        assert node.text == passages.format_passage(passage)
        assert node.metadata == passage.metadata()
        assert node.node_id == passage.id
        assert item.score == passage.score


def test_consolidated_hit_names_the_act_to_cite() -> None:
    retriever, _ = leg_retriever("legislation.consolidated.json", as_of=AS_OF)
    (item,) = retriever.retrieve("q")
    meta = text_node(item).metadata
    assert meta["consolidated"] is True
    assert meta["celex"] == "02099R0001-20990101"
    assert meta["cite_celex"] == "32099R0001"
    assert "no legal force" in str(meta["citation"])
    assert "Source: CELEX 32099R0001" in text_node(item).text
    assert item.score == 0.91


def test_no_date_means_no_as_of_keys() -> None:
    retriever, _ = leg_retriever("legislation.no-date.json")
    for item in retriever.retrieve("q"):
        assert not [k for k in text_node(item).metadata if k.startswith("as_of")]


def test_resolved_hit_carries_its_version() -> None:
    retriever, _ = leg_retriever("legislation.as-published.json", as_of=AS_OF)
    (item,) = retriever.retrieve("q")
    meta = text_node(item).metadata
    assert meta["as_of"] == AS_OF
    assert meta["as_of_resolved"] is True
    assert meta["version_kind"] == "as_published"
    assert "version for 2099-06-01: 32099R0002" in text_node(item).text


def test_unresolved_hit_is_withheld_by_default() -> None:
    retriever, _ = leg_retriever("legislation.unresolved.json", as_of=AS_OF)
    got = retriever.retrieve_with_result("q")
    (item,) = got.nodes
    node = text_node(item)
    assert node.text.startswith("[No text: CELEX 32099L0003 art_12 is not established")
    assert "fetch_timeout" in node.text
    assert node.metadata["text_withheld"] is True
    assert node.metadata["text_status"] == "withheld"
    assert item.score == 0.75
    for view in all_views(item):
        assert "UNRESOLVEDTEXTTOKEN" not in view
        assert "UNRESOLVEDSNIPPETTOKEN" not in view
    assert got.result.withheld == 1


def test_drop_and_keep_labelled() -> None:
    dropped, _ = leg_retriever("legislation.unresolved.json", as_of=AS_OF, unresolved="drop")
    got = dropped.retrieve_with_result("q")
    assert got.nodes == []
    assert got.result.dropped == 1
    kept, _ = leg_retriever("legislation.unresolved.json", as_of=AS_OF, unresolved="keep_labelled")
    (item,) = kept.retrieve("q")
    assert text_node(item).text.startswith("[Latest indexed text of CELEX 32099L0003 art_12;")
    assert "UNRESOLVEDTEXTTOKEN" in text_node(item).text
    assert text_node(item).metadata["text_status"] == "not_established_for_date"


def test_emit_withheld_false_leaves_withheld_nodes_out_but_counts_them() -> None:
    retriever, _ = leg_retriever("legislation.mixed.json", as_of=AS_OF, emit_withheld=False)
    got = retriever.retrieve_with_result("q")
    assert [text_node(n).metadata["celex"] for n in got.nodes] == [
        "02099R0001-20990101",
        "32099R0002",
    ]
    assert got.result.withheld == 1


def test_the_retrievers_date_is_checked_on_every_hit() -> None:
    """A hit marked resolved for another date than the one asked is not
    resolved for the date asked."""
    body = load_fixture("legislation.as-published.json")
    body["results"][0]["asOf"] = "2099-01-01"
    body["asOf"] = "2099-01-01"
    client, _ = make_client({LEG: Route(body)})
    (item,) = LexAPILegislationRetriever(client, as_of=AS_OF).retrieve("q")
    assert text_node(item).metadata["text_withheld"] is True
    assert "as published article five" not in text_node(item).text


def test_include_citation_in_text_false_uses_content() -> None:
    retriever, _ = leg_retriever(
        "legislation.mixed.json", as_of=AS_OF, include_citation_in_text=False
    )
    expected = leg_expected("legislation.mixed.json", as_of=AS_OF).passages
    texts = [text_node(n).text for n in retriever.retrieve("q")]
    assert texts == [p.content() for p in expected]
    assert all("Source:" not in t for t in texts)


def test_empty_response_is_an_empty_list() -> None:
    retriever, _ = leg_retriever("legislation.empty.json")
    got = retriever.retrieve_with_result("q")
    assert got.nodes == []
    assert got.result.passages == ()
    assert got.result.units_charged == 5


def test_metadata_values_are_flat_scalars() -> None:
    retriever, _ = leg_retriever("legislation.mixed.json", as_of=AS_OF)
    for item in retriever.retrieve("q"):
        for value in text_node(item).metadata.values():
            assert isinstance(value, str | int | float | bool)


def test_ids_are_stable_and_differ_by_date() -> None:
    first, _ = leg_retriever("legislation.mixed.json", as_of=AS_OF)
    second, _ = leg_retriever("legislation.mixed.json", as_of=AS_OF)
    ids1 = [n.node.node_id for n in first.retrieve("q")]
    ids2 = [n.node.node_id for n in second.retrieve("q")]
    assert ids1 == ids2
    undated, _ = leg_retriever("legislation.no-date.json")
    undated_ids = {n.node.node_id for n in undated.retrieve("q")}
    assert undated_ids.isdisjoint(ids1)


# ── what the LLM and the embedder see ──────────────────────────────────


def test_excluded_metadata_keys_are_pinned() -> None:
    assert EXCLUDED_LLM_METADATA_KEYS == ("eli", "id", "kind", "score", "url")
    retriever, _ = leg_retriever("legislation.mixed.json", as_of=AS_OF)
    for item in retriever.retrieve("q"):
        node = text_node(item)
        assert node.excluded_embed_metadata_keys == sorted(node.metadata)
        assert node.excluded_llm_metadata_keys == ["eli", "id", "kind", "score", "url"]
        assert not set(GUARDRAIL_KEYS) & set(node.excluded_llm_metadata_keys)


def test_llm_view_shows_the_guardrail_keys() -> None:
    retriever, _ = leg_retriever("legislation.mixed.json", as_of=AS_OF)
    for item in retriever.retrieve("q"):
        node = text_node(item)
        llm = node.get_content(metadata_mode=MetadataMode.LLM)
        for key in GUARDRAIL_KEYS:
            if key in node.metadata:
                assert f"{key}: {node.metadata[key]}" in llm
        for key in EXCLUDED_LLM_METADATA_KEYS:
            assert f"\n{key}: " not in llm and not llm.startswith(f"{key}: ")


def test_embed_view_holds_no_metadata() -> None:
    retriever, _ = leg_retriever("legislation.mixed.json", as_of=AS_OF)
    for item in retriever.retrieve("q"):
        node = text_node(item)
        assert node.get_content(metadata_mode=MetadataMode.EMBED) == node.text


# ── errors ──────────────────────────────────────────────────────────────

ERRORS = [
    ("error.401.json", 401, lexapi.AuthenticationError, {}),
    ("error.402.json", 402, lexapi.CreditsExhaustedError, {}),
    ("error.403-past-due.json", 403, lexapi.PaymentPastDueError, {}),
    ("error.429.json", 429, lexapi.RateLimitedError, {"Retry-After": "30"}),
]


@pytest.mark.parametrize(("fixture", "status", "error", "headers"), ERRORS)
def test_errors_propagate(
    fixture: str, status: int, error: type[Exception], headers: dict[str, str]
) -> None:
    routes = {LEG: fixture_route(fixture, status, headers), CASE: fixture_route(fixture, status)}
    client, _ = make_client(routes)
    for retriever in (LexAPILegislationRetriever(client), LexAPICaseLawRetriever(client)):
        with pytest.raises(error):
            retriever.retrieve("q")
        with pytest.raises(error):
            retriever.retrieve_with_result("q")


@pytest.mark.parametrize(("fixture", "status", "error", "headers"), ERRORS)
async def test_errors_propagate_async(
    fixture: str, status: int, error: type[Exception], headers: dict[str, str]
) -> None:
    routes = {LEG: fixture_route(fixture, status, headers), CASE: fixture_route(fixture, status)}
    aclient, _ = make_async_client(routes)
    for retriever in (
        LexAPILegislationRetriever(async_client=aclient),
        LexAPICaseLawRetriever(async_client=aclient),
    ):
        with pytest.raises(error):
            await retriever.aretrieve("q")
        with pytest.raises(error):
            await retriever.aretrieve_with_result("q")


def test_past_due_error_keeps_its_details() -> None:
    client, _ = make_client({LEG: fixture_route("error.403-past-due.json", 403)})
    with pytest.raises(lexapi.PaymentPastDueError) as info:
        LexAPILegislationRetriever(client).retrieve("q")
    assert info.value.billing_url == "https://lex-api.com/dashboard#billing"


# ── sync == async ───────────────────────────────────────────────────────


def snapshot(nodes: list[NodeWithScore]) -> list[tuple[Any, ...]]:
    return [(n.node.node_id, text_node(n).text, text_node(n).metadata, n.score) for n in nodes]


@pytest.mark.parametrize(("fixture", "as_of"), FIXTURES)
async def test_async_equals_sync(fixture: str, as_of: str | None) -> None:
    client, sync_rec = make_client({LEG: fixture_route(fixture)})
    aclient, async_rec = make_async_client({LEG: fixture_route(fixture)})
    retriever = LexAPILegislationRetriever(client, aclient, as_of=as_of)
    sync_nodes = retriever.retrieve("q")
    async_nodes = await retriever.aretrieve("q")
    assert snapshot(sync_nodes) == snapshot(async_nodes)
    assert [r.content for r in sync_rec.requests] == [r.content for r in async_rec.requests]


async def test_async_with_only_a_sync_client_runs_in_a_thread() -> None:
    retriever, requests = leg_retriever("legislation.mixed.json", as_of=AS_OF)
    got = await retriever.aretrieve_with_result("q")
    assert len(got.nodes) == 3
    assert len(requests) == 1
    assert got.result.units_charged == 9


def test_sync_with_only_an_async_client_says_so() -> None:
    aclient, _ = make_async_client({})
    retriever = LexAPILegislationRetriever(async_client=aclient)
    with pytest.raises(TypeError, match="only an async_client"):
        retriever.retrieve("q")


async def test_async_case_law_equals_sync() -> None:
    client, _ = make_client({CASE: fixture_route("caselaw.with-text.json")})
    aclient, _ = make_async_client({CASE: fixture_route("caselaw.with-text.json")})
    retriever = LexAPICaseLawRetriever(client, aclient, max_chars=60)
    assert snapshot(retriever.retrieve("q")) == snapshot(await retriever.aretrieve("q"))


# ── credits: per call, concurrency-safe ────────────────────────────────


def test_retrieve_with_result_reports_credits() -> None:
    retriever, _ = leg_retriever("legislation.mixed.json", as_of=AS_OF)
    got = retriever.retrieve_with_result("q")
    assert got.result.units_charged == 9
    assert got.result.credits_remaining == 991
    assert got.result.withheld == 1
    assert snapshot(got.nodes) == snapshot(retriever.retrieve("q"))


def test_no_result_state_is_kept_on_the_retriever() -> None:
    retriever, _ = leg_retriever("legislation.mixed.json", as_of=AS_OF)
    before = dict(vars(retriever))
    retriever.retrieve_with_result("q")
    assert dict(vars(retriever)).keys() == before.keys()
    assert not any(isinstance(v, PassageResult) for v in vars(retriever).values())


def two_retrievers_routes() -> tuple[dict[str, Route], dict[str, Route]]:
    one = load_fixture("legislation.mixed.json")
    two = load_fixture("legislation.no-date.json")
    one["credits"] = {"operation_weight": 11, "remaining": 1}
    two["credits"] = {"operation_weight": 22, "remaining": 2}
    return {LEG: Route(one)}, {LEG: Route(two)}


async def test_concurrent_async_calls_keep_their_own_results() -> None:
    routes_one, routes_two = two_retrievers_routes()
    a1, _ = make_async_client(routes_one)
    a2, _ = make_async_client(routes_two)
    r1 = LexAPILegislationRetriever(async_client=a1, as_of=AS_OF)
    r2 = LexAPILegislationRetriever(async_client=a2)
    got = await asyncio.gather(
        *[(r1 if i % 2 else r2).aretrieve_with_result("q") for i in range(20)]
    )
    for i, item in enumerate(got):
        assert item.result.units_charged == (11 if i % 2 else 22)


def test_concurrent_threads_keep_their_own_results() -> None:
    routes_one, routes_two = two_retrievers_routes()
    c1, _ = make_client(routes_one)
    c2, _ = make_client(routes_two)
    shared_one = LexAPILegislationRetriever(c1, as_of=AS_OF)
    shared_two = LexAPILegislationRetriever(c2)
    seen: list[tuple[int, int | None]] = []
    lock = threading.Lock()

    def work(i: int) -> None:
        retriever = shared_one if i % 2 else shared_two
        units = retriever.retrieve_with_result("q").result.units_charged
        with lock:
            seen.append((i, units))

    threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen) == [(i, 11 if i % 2 else 22) for i in range(16)]


class _Collector(BaseEventHandler):
    events: list[BaseEvent] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "Collector"

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        if isinstance(event, LexAPIRetrievalEvent):
            self.events.append(event)


def test_each_retrieval_dispatches_its_result() -> None:
    collector = _Collector(events=[])
    root = get_dispatcher()
    root.add_event_handler(collector)
    try:
        retriever, _ = leg_retriever("legislation.mixed.json", as_of=AS_OF)
        retriever.retrieve("transfer")
    finally:
        root.event_handlers.remove(collector)
    (event,) = collector.events
    assert isinstance(event, LexAPIRetrievalEvent)
    assert event.retriever == "LexAPILegislationRetriever"
    assert event.query == "transfer"
    assert event.result.units_charged == 9
    assert event.span_id is not None


# ── clients and secrets ────────────────────────────────────────────────


def test_client_and_key_together_are_refused() -> None:
    client, _ = make_client({})
    with pytest.raises(ValueError, match="not both"):
        LexAPILegislationRetriever(client, api_key="lex_x")
    with pytest.raises(TypeError):
        LexAPILegislationRetriever(client="not a client")  # type: ignore[arg-type]


def test_no_key_anywhere_is_an_sdk_error() -> None:
    with pytest.raises(ValueError, match="api_key"):
        LexAPILegislationRetriever()


SECRET = "lex_secret_key_must_not_leak_42"


def test_owned_clients_carry_the_suffix_and_hide_the_key() -> None:
    retriever = LexAPILegislationRetriever(api_key=SECRET, base_url=LOOPBACK)
    agent = retriever._clients.sync._http.headers["user-agent"]
    assert agent.endswith(user_agent_suffix())
    assert user_agent_suffix().startswith("llama-index-retrievers-lexapi/")
    for text in (repr(retriever), str(retriever), repr(vars(retriever)), repr(retriever._clients)):
        assert SECRET not in text
    retriever.close()


def test_owned_clients_cannot_reach_the_network() -> None:
    retriever = LexAPILegislationRetriever(api_key=SECRET, base_url=LOOPBACK)
    with pytest.raises(RuntimeError, match="Offline tests blocked"):
        retriever.retrieve("q")


def test_the_default_base_url_cannot_be_reached() -> None:
    retriever = LexAPICaseLawRetriever(api_key=SECRET)
    assert str(retriever._clients.sync._http.base_url).startswith("https://")
    with pytest.raises(RuntimeError, match="Offline tests blocked"):
        retriever.retrieve("q")


async def test_owned_async_client_cannot_reach_the_network() -> None:
    retriever = LexAPILegislationRetriever(api_key=SECRET, base_url=LOOPBACK)
    with pytest.raises(RuntimeError, match="Offline tests blocked"):
        await retriever.aretrieve("q")
    await retriever.aclose()


def test_passed_clients_are_not_closed() -> None:
    retriever, _ = leg_retriever("legislation.empty.json")
    retriever.close()
    assert retriever.retrieve("q") == []


# ── case law ────────────────────────────────────────────────────────────


def case_retriever(fixture: str, **kwargs: Any) -> tuple[LexAPICaseLawRetriever, list[Any]]:
    client, recorder = make_client({CASE: fixture_route(fixture)})
    return LexAPICaseLawRetriever(client, **kwargs), recorder.requests


def test_case_law_request_asks_for_text_and_sends_filters() -> None:
    import datetime as dt

    retriever, requests = case_retriever(
        "caselaw.with-text.json",
        court="Court of Justice",
        date_from=dt.date(2090, 1, 1),
        date_to="2099-12-31",
        similarity_top_k=4,
    )
    retriever.retrieve("q")
    assert json.loads(requests[0].content) == {
        "query": "q",
        "limit": 4,
        "include": ["text"],
        "filters": {
            "court": "Court of Justice",
            "date_from": "2090-01-01",
            "date_to": "2099-12-31",
        },
    }


def test_case_law_never_sends_hyde() -> None:
    retriever, requests = case_retriever("caselaw.with-text.json")
    retriever.retrieve("q")
    assert "hyde" not in json.loads(requests[0].content)


def test_case_law_nodes_are_the_core_passages() -> None:
    retriever, _ = case_retriever("caselaw.with-text.json")
    nodes = retriever.retrieve("q")
    expected = passages.from_case_law_search(
        SemanticSearchResponse(load_fixture("caselaw.with-text.json"))
    ).passages
    assert [text_node(n).text for n in nodes] == [passages.format_passage(p) for p in expected]
    assert [text_node(n).metadata for n in nodes] == [p.metadata() for p in expected]
    assert [n.score for n in nodes] == [p.score for p in expected]
    assert text_node(nodes[0]).metadata["ecli"] == "ECLI:EU:C:2099:1"
    assert "para. 93" in text_node(nodes[0]).text


@pytest.mark.parametrize("fixture", ["caselaw.with-text.json", "caselaw.without-text.json"])
def test_case_law_never_uses_snippet_or_summary(fixture: str) -> None:
    retriever, _ = case_retriever(fixture)
    for item in retriever.retrieve("q"):
        for view in all_views(item):
            assert "SNIPPETTOKEN" not in view
            assert "SUMMARYTOKEN" not in view


def test_case_law_without_text_gives_a_notice() -> None:
    retriever, requests = case_retriever("caselaw.without-text.json", include_text=False)
    nodes = retriever.retrieve("q")
    assert "include" not in json.loads(requests[0].content)
    assert text_node(nodes[0]).text.startswith("[No text returned for CELEX 62099CJ9901.]")
    assert text_node(nodes[0]).metadata["text_status"] == "absent"


def test_case_law_max_chars_cuts_on_a_paragraph_break_and_marks_it() -> None:
    retriever, _ = case_retriever("caselaw.with-text.json", max_chars=110)
    first = text_node(retriever.retrieve("q")[0]).text
    body = load_fixture("caselaw.with-text.json")["results"][0]["text"]
    paragraphs = body.split("\n\n")
    assert first.startswith(paragraphs[0] + "\n[Text cut by the retriever after ")
    assert paragraphs[1] not in first
    assert f"of {len(body)} characters (max_chars=110)" in first
    assert "\nSource: CELEX 62099CJ9901" in first


def test_case_law_max_chars_leaves_short_texts_alone() -> None:
    retriever, _ = case_retriever("caselaw.with-text.json", max_chars=100_000)
    expected = passages.from_case_law_search(
        SemanticSearchResponse(load_fixture("caselaw.with-text.json"))
    ).passages
    assert [text_node(n).text for n in retriever.retrieve("q")] == [
        passages.format_passage(p) for p in expected
    ]


@pytest.mark.parametrize("bad", [0, -5, True, 1.5])
def test_case_law_max_chars_is_validated(bad: Any) -> None:
    client, _ = make_client({})
    with pytest.raises(ValueError, match="max_chars"):
        LexAPICaseLawRetriever(client, max_chars=bad)


def test_case_law_duplicate_ids_are_merged() -> None:
    body = load_fixture("caselaw.with-text.json")
    twin = dict(body["results"][1])
    twin["score"] = 0.5
    body["results"].append(twin)
    client, _ = make_client({CASE: Route(body)})
    got = LexAPICaseLawRetriever(client).retrieve_with_result("q")
    assert len(got.result.passages) == 3
    assert [n.score for n in got.nodes] == [0.83, 0.7]


def test_case_law_credits() -> None:
    retriever, _ = case_retriever("caselaw.with-text.json")
    got = retriever.retrieve_with_result("q")
    assert (got.result.units_charged, got.result.credits_remaining) == (5, 995)
