"""Behaviour of the LexAPI tool spec, offline, against synthetic fixtures."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import textwrap
from typing import Any

import lexapi
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from lexapi import passages
from lexapi.models import DocumentResponse, LegislationAsOfResponse, SemanticLegislationResponse
from lexapi_offline import (
    LOOPBACK,
    Route,
    fixture_route,
    legislation_body,
    legislation_hit,
    load_fixture,
    make_async_client,
    make_client,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.calling import acall_tool, call_tool
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.lexapi import LexAPIToolError, LexAPIToolEvent, LexAPIToolSpec

LEG = "/legislation/semantic"
DOC = "/documentContent"
ASOF = "/legislation/as-of"
AS_OF = "2099-06-01"


def spec_for(routes: dict[str, Route], **kwargs: Any) -> tuple[LexAPIToolSpec, list[Any]]:
    client, recorder = make_client(routes)
    return LexAPIToolSpec(client, **kwargs), recorder.requests


def tools_by_name(spec: LexAPIToolSpec) -> dict[str, FunctionTool]:
    return {t.metadata.name or "": t for t in spec.to_tool_list()}


# ── shape: names, descriptions, signatures ─────────────────────────────

SEARCH_DOC = "Search EU legislation by meaning; returns matching articles, each with its citation."
READ_DOC = "Read an EU document by its CELEX number: one article, or every article."
DATE_DOC = "The version of an EU act that LexAPI lists for a date, and optionally one provision."
PARAM_DOCS = {
    "query": "What to look for, in plain language.",
    "as_of": (
        "Optional date, YYYY-MM-DD. Each article is then given in the version LexAPI lists "
        "for that date; when that could not be established the article is returned as a "
        "notice without text, with the reason."
    ),
    "celex": None,
    "article": (
        "Optional article to read, e.g. art_17 or 17. Without it every article of the "
        "document is returned, which can be long."
    ),
    "language": "Optional language code of the text, e.g. en, de, fr.",
    "date": "The date, YYYY-MM-DD.",
    "provision": (
        "Optional provision, e.g. art_17, anx_III or rct_26. Without it only the version is named."
    ),
}

EXPECTED_TOOLS = {
    "lexapi_search_legislation": (
        "(query: str, as_of: str | None = None) -> str",
        SEARCH_DOC,
        {"query": "string", "as_of": None},
        ["query"],
    ),
    "lexapi_read_document": (
        "(celex: str, article: str | None = None, language: str | None = None) -> str",
        READ_DOC,
        {"celex": "string", "article": None, "language": None},
        ["celex"],
    ),
    "lexapi_law_on_date": (
        "(celex: str, date: str, provision: str | None = None, language: str | None = None) -> str",
        DATE_DOC,
        {"celex": "string", "date": "string", "provision": None, "language": None},
        ["celex", "date"],
    ),
}


def test_spec_derives_from_base_tool_spec() -> None:
    assert BaseToolSpec in LexAPIToolSpec.__mro__


def test_spec_functions_are_pinned() -> None:
    assert LexAPIToolSpec.spec_functions == [
        ("lexapi_search_legislation", "alexapi_search_legislation"),
        ("lexapi_read_document", "alexapi_read_document"),
        ("lexapi_law_on_date", "alexapi_law_on_date"),
    ]


def test_tool_names_descriptions_and_signatures_are_pinned() -> None:
    spec, _ = spec_for({})
    tools = tools_by_name(spec)
    assert list(tools) == list(EXPECTED_TOOLS)
    for name, (signature, doc, props, required) in EXPECTED_TOOLS.items():
        tool = tools[name]
        assert str(inspect.signature(getattr(spec, name))) == signature
        assert str(inspect.signature(getattr(spec, "a" + name))) == signature
        assert tool.metadata.description == f"{name}{signature}\n{doc}"
        params = tool.metadata.get_parameters_dict()
        assert params["required"] == required
        assert list(params["properties"]) == list(props)
        for key, kind in props.items():
            schema = params["properties"][key]
            if PARAM_DOCS[key] is not None:
                assert schema["description"] == PARAM_DOCS[key]
            else:
                assert schema["description"].startswith("CELEX number of the ")
            default = inspect.signature(getattr(spec, name)).parameters[key].default
            if default is inspect.Parameter.empty:
                assert "default" not in schema
            else:
                assert schema["default"] == default
            if kind is None:
                assert schema["anyOf"] == [{"type": "string"}, {"type": "null"}]
                assert schema["default"] is None
            else:
                assert schema["type"] == kind


def test_every_tool_has_a_real_async_variant() -> None:
    spec, _ = spec_for({})
    for name in EXPECTED_TOOLS:
        assert inspect.iscoroutinefunction(getattr(spec, "a" + name))
        assert not inspect.iscoroutinefunction(getattr(spec, name))


def test_descriptions_make_no_compliance_or_advice_claims() -> None:
    spec, _ = spec_for({})
    text = " ".join(t.metadata.description for t in spec.to_tool_list()).lower()
    for word in ("complian", "legal advice", "in force", "guarantee", "accurate", "official"):
        assert word not in text


# ── search ──────────────────────────────────────────────────────────────


def expected_search(fixture: str, as_of: str | None, policy: Any = "withhold") -> str:
    result = passages.from_legislation_search(
        SemanticLegislationResponse(load_fixture(fixture)), unresolved=policy, as_of=as_of
    )
    return "\n\n".join(passages.format_passage(p) for p in result.passages)


def test_search_request_and_output() -> None:
    spec, requests = spec_for({LEG: fixture_route("legislation.mixed.json")}, similarity_top_k=3)
    out = spec.lexapi_search_legislation("erasure", as_of=AS_OF)
    assert json.loads(requests[0].content) == {
        "query": "erasure",
        "limit": 3,
        "include": ["text"],
        "asOf": AS_OF,
    }
    assert out == expected_search("legislation.mixed.json", AS_OF)
    assert "UNRESOLVEDTEXTTOKEN" not in out
    assert "UNRESOLVEDSNIPPETTOKEN" not in out
    assert "[No text: CELEX 32099L0003 art_12" in out
    assert "Source: CELEX 32099R0001, art_17, consolidated text" in out


def test_search_without_a_date() -> None:
    spec, requests = spec_for({LEG: fixture_route("legislation.no-date.json")})
    out = spec.lexapi_search_legislation("erasure")
    assert "asOf" not in json.loads(requests[0].content)
    assert out == expected_search("legislation.no-date.json", None)


def test_search_checks_the_date_on_every_hit() -> None:
    body = load_fixture("legislation.as-published.json")
    body["results"][0]["asOf"] = "2099-01-01"
    spec, _ = spec_for({LEG: Route(body)})
    out = spec.lexapi_search_legislation("q", as_of=AS_OF)
    assert "as published article five" not in out
    assert out.startswith("[No text: CELEX 32099R0002 art_5")


def test_search_drop_policy_says_what_was_left_out() -> None:
    spec, _ = spec_for({LEG: fixture_route("legislation.unresolved.json")}, unresolved="drop")
    out = spec.lexapi_search_legislation("q", as_of=AS_OF)
    assert out == (
        "LexAPI returned no matching articles.\n\n"
        "1 hit(s) left out: text for 2099-06-01 not established."
    )


def test_search_empty_and_hint() -> None:
    body = load_fixture("legislation.empty.json")
    spec, _ = spec_for({LEG: Route(body)})
    assert spec.lexapi_search_legislation("q") == "LexAPI returned no matching articles."
    body["hint"] = "Few results; try other words."
    spec, _ = spec_for({LEG: Route(body)})
    assert spec.lexapi_search_legislation("q").endswith(
        "\n\nHint from LexAPI: Few results; try other words."
    )


# ── read a document ─────────────────────────────────────────────────────


def test_read_one_article() -> None:
    spec, requests = spec_for({DOC: fixture_route("document.read.json")})
    out = spec.lexapi_read_document("32099R0001", article="art_17", language="de")
    assert json.loads(requests[0].content) == {
        "celexNumber": "32099R0001",
        "language": "de",
        "include": ["metadata", "articles"],
        "articleId": "art_17",
    }
    result = passages.from_document(DocumentResponse(load_fixture("document.read.json")))
    assert out == "\n\n".join(passages.format_passage(p) for p in result.passages)


def test_read_whole_document() -> None:
    spec, requests = spec_for({DOC: fixture_route("document.read.json")}, language="fr")
    out = spec.lexapi_read_document("32099R0001")
    assert json.loads(requests[0].content) == {"celexNumber": "32099R0001", "language": "fr"}
    assert out.count("Source: CELEX 32099R0001") == 2


def test_read_document_without_text() -> None:
    body = load_fixture("document.read.json")
    body["document"]["content"] = None
    spec, _ = spec_for({DOC: Route(body)})
    assert (
        spec.lexapi_read_document("32099R0001", article="art_9")
        == "LexAPI returned no text for CELEX 32099R0001 art_9."
    )


# ── law on a date ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fixture", "provision", "starts"),
    [
        ("as-of.resolved.json", "art_17", "[SYNTHETIC FIXTURE"),
        ("as-of.unresolved.json", "art_17", "[No text: CELEX 32099R0001 art_17 is not established"),
        ("as-of.version-only.json", None, "[Version 02099R0001-20990101 (consolidated) is the"),
    ],
)
def test_law_on_date(fixture: str, provision: str | None, starts: str) -> None:
    spec, requests = spec_for({ASOF: fixture_route(fixture)})
    out = spec.lexapi_law_on_date("32099R0001", AS_OF, provision=provision)
    sent = {"document": "32099R0001", "asOf": AS_OF}
    if provision:
        sent["provision"] = provision
    assert json.loads(requests[0].content) == sent
    result = passages.from_legislation_as_of(
        LegislationAsOfResponse(load_fixture(fixture)), provision=provision
    )
    assert out == passages.format_passage(result.passages[0])
    assert out.startswith(starts)


def test_law_on_date_resolved_names_the_act_to_cite() -> None:
    spec, _ = spec_for({ASOF: fixture_route("as-of.resolved.json")})
    out = spec.lexapi_law_on_date("32099R0001", AS_OF, provision="art_17")
    assert (
        "Source: CELEX 32099R0001, art_17, consolidated text 02099R0001-20990101 (no legal force)"
        in out
    )
    assert "version for 2099-06-01: 02099R0001-20990101" in out


def test_law_on_date_unresolved_drop() -> None:
    spec, _ = spec_for({ASOF: fixture_route("as-of.unresolved.json")}, unresolved="drop")
    out = spec.lexapi_law_on_date("32099R0001", AS_OF, provision="art_17")
    assert out.startswith("LexAPI returned no answer for CELEX 32099R0001.")
    assert "1 hit(s) left out" in out


# ── errors ──────────────────────────────────────────────────────────────

ERRORS = [
    ("error.401.json", 401, "UNAUTHORIZED", lexapi.AuthenticationError, {}),
    ("error.402.json", 402, "CREDITS_EXHAUSTED", lexapi.CreditsExhaustedError, {}),
    ("error.403-past-due.json", 403, "PAYMENT_PAST_DUE", lexapi.PaymentPastDueError, {}),
    ("error.429.json", 429, "RATE_LIMITED", lexapi.RateLimitedError, {"Retry-After": "30"}),
]

CALLS = [
    ("lexapi_search_legislation", {"query": "q"}),
    ("lexapi_read_document", {"celex": "32099R0001"}),
    ("lexapi_law_on_date", {"celex": "32099R0001", "date": AS_OF}),
]


def error_routes(fixture: str, status: int, headers: dict[str, str]) -> dict[str, Route]:
    return {path: fixture_route(fixture, status, headers) for path in (LEG, DOC, ASOF)}


@pytest.mark.parametrize(("fixture", "status", "code", "sdk_error", "headers"), ERRORS)
def test_errors_raise_with_code_and_message(
    fixture: str, status: int, code: str, sdk_error: type[Exception], headers: dict[str, str]
) -> None:
    spec, _ = spec_for(error_routes(fixture, status, headers))
    message = load_fixture(fixture)["error"]["message"]
    for name, args in CALLS:
        with pytest.raises(LexAPIToolError) as info:
            getattr(spec, name)(**args)
        assert info.value.code == code
        assert info.value.status == status
        assert str(info.value).startswith(f"LexAPI error {code} (HTTP {status}): {message}")
        assert isinstance(info.value.__cause__, sdk_error)
        if status == 429:
            assert info.value.retry_after == 30
            assert str(info.value).endswith("Retry after 30 s.")


@pytest.mark.parametrize(("fixture", "status", "code", "sdk_error", "headers"), ERRORS)
async def test_errors_raise_async(
    fixture: str, status: int, code: str, sdk_error: type[Exception], headers: dict[str, str]
) -> None:
    aclient, _ = make_async_client(error_routes(fixture, status, headers))
    spec = LexAPIToolSpec(async_client=aclient)
    for name, args in CALLS:
        with pytest.raises(LexAPIToolError) as info:
            await getattr(spec, "a" + name)(**args)
        assert info.value.code == code
        assert isinstance(info.value.__cause__, sdk_error)


@pytest.mark.parametrize(("fixture", "status", "code", "sdk_error", "headers"), ERRORS)
async def test_errors_reach_the_agent_as_error_output(
    fixture: str, status: int, code: str, sdk_error: type[Exception], headers: dict[str, str]
) -> None:
    """LlamaIndex's tool calling turns a raised error into ``is_error``
    output whose content is the error message."""
    spec, _ = spec_for(error_routes(fixture, status, headers))
    tools = tools_by_name(spec)
    for name, args in CALLS:
        for output in (call_tool(tools[name], args), await acall_tool(tools[name], args)):
            assert output.is_error is True
            assert f"LexAPI error {code} (HTTP {status})" in output.content
            assert "lex_offline_test_key" not in output.content


def test_error_text_masks_anything_shaped_like_a_key() -> None:
    error = lexapi.LexAPIError("key lex_abcdef123456 rejected", code="UNAUTHORIZED", status=401)
    wrapped = LexAPIToolError.from_sdk(error)
    assert "lex_abcdef123456" not in str(wrapped)
    assert "lex_***" in str(wrapped)
    assert "lex_abcdef123456" not in repr(wrapped)


# ── sync == async ───────────────────────────────────────────────────────

ROUTES = {
    LEG: fixture_route("legislation.mixed.json"),
    DOC: fixture_route("document.read.json"),
    ASOF: fixture_route("as-of.resolved.json"),
}
OK_CALLS = [
    ("lexapi_search_legislation", {"query": "q", "as_of": AS_OF}),
    ("lexapi_read_document", {"celex": "32099R0001", "article": "art_17"}),
    ("lexapi_law_on_date", {"celex": "32099R0001", "date": AS_OF, "provision": "art_17"}),
]


async def test_async_equals_sync() -> None:
    client, sync_rec = make_client(ROUTES)
    aclient, async_rec = make_async_client(ROUTES)
    spec = LexAPIToolSpec(client, aclient)
    for name, args in OK_CALLS:
        assert getattr(spec, name)(**args) == await getattr(spec, "a" + name)(**args)
    assert [r.content for r in sync_rec.requests] == [r.content for r in async_rec.requests]


async def test_async_with_only_a_sync_client_runs_in_a_thread() -> None:
    spec, requests = spec_for(ROUTES)
    for name, args in OK_CALLS:
        assert getattr(spec, name)(**args) == await getattr(spec, "a" + name)(**args)
    assert len(requests) == 6


async def test_function_tools_call_sync_and_async() -> None:
    spec, _ = spec_for(ROUTES)
    tools = tools_by_name(spec)
    for name, args in OK_CALLS:
        sync_out = tools[name].call(**args)
        async_out = await tools[name].acall(**args)
        assert sync_out.content == async_out.content == getattr(spec, name)(**args)
        assert sync_out.is_error is False


def test_sync_with_only_an_async_client_says_so() -> None:
    aclient, _ = make_async_client({})
    spec = LexAPIToolSpec(async_client=aclient)
    with pytest.raises(TypeError, match="only an async_client"):
        spec.lexapi_search_legislation("q")


# ── credits ─────────────────────────────────────────────────────────────


class _Collector(BaseEventHandler):
    events: list[BaseEvent] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "Collector"

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        if isinstance(event, LexAPIToolEvent):
            self.events.append(event)


def test_each_call_dispatches_its_result() -> None:
    collector = _Collector(events=[])
    root = get_dispatcher()
    root.add_event_handler(collector)
    try:
        spec, _ = spec_for(ROUTES)
        for name, args in OK_CALLS:
            getattr(spec, name)(**args)
    finally:
        root.event_handlers.remove(collector)
    got = [
        (e.tool, e.result.units_charged) for e in collector.events if isinstance(e, LexAPIToolEvent)
    ]
    assert got == [
        ("lexapi_search_legislation", 9),
        ("lexapi_read_document", 1),
        ("lexapi_law_on_date", 4),
    ]


# ── clients and secrets ────────────────────────────────────────────────

SECRET = "lex_secret_key_must_not_leak_42"


def test_validation() -> None:
    client, _ = make_client({})
    with pytest.raises(ValueError, match="similarity_top_k"):
        LexAPIToolSpec(client, similarity_top_k=0)
    with pytest.raises(ValueError, match="similarity_top_k"):
        LexAPIToolSpec(client, similarity_top_k=51)
    with pytest.raises(ValueError, match="unresolved"):
        LexAPIToolSpec(client, unresolved="latest_labelled")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not both"):
        LexAPIToolSpec(client, api_key="lex_x")


def test_key_is_not_in_repr_and_owned_clients_cannot_reach_the_network() -> None:
    spec = LexAPIToolSpec(api_key=SECRET, base_url=LOOPBACK)
    for text in (repr(spec), str(spec), repr(vars(spec))):
        assert SECRET not in text
    assert spec._clients.sync._http.headers["user-agent"].endswith("llama-index-tools-lexapi/0.1.0")
    with pytest.raises(RuntimeError, match="Offline tests blocked"):
        spec.lexapi_search_legislation("q")
    spec.close()


def test_default_base_url_cannot_be_reached() -> None:
    spec = LexAPIToolSpec(api_key=SECRET)
    with pytest.raises(RuntimeError, match="Offline tests blocked"):
        spec.lexapi_law_on_date("32099R0001", AS_OF)


IMPORT_PROBE = textwrap.dedent(
    """
    import socket, sys
    def refuse(*a, **k):
        raise SystemExit("network at import")
    socket.socket.connect = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse
    sys.modules["openai"] = None
    import llama_index.tools.lexapi as m
    assert "openai" not in [k for k, v in sys.modules.items() if v is not None]
    print(sorted(m.__all__))
    """
)


def test_import_needs_no_openai_and_no_network() -> None:
    out = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "LexAPIToolSpec" in out.stdout


# ── property: no unresolved token in the search tool's output ──────────


@settings(max_examples=500, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    spec_list=st.lists(
        st.sampled_from(["resolved", "unresolved", "other_date", "undated"]), max_size=8
    ),
    policy=st.sampled_from(["withhold", "drop"]),
)
def test_search_output_never_carries_unresolved_text(spec_list: list[str], policy: Any) -> None:
    hits = []
    forbidden: set[str] = set()
    for n, kind in enumerate(spec_list):
        if kind == "resolved":
            hits.append(legislation_hit(n, resolved=True))
            continue
        if kind == "unresolved":
            hits.append(legislation_hit(n, resolved=False))
        elif kind == "other_date":
            hits.append(legislation_hit(n, resolved=True, echoed_as_of="2098-01-01"))
        else:
            hits.append(legislation_hit(n, resolved=None))
        forbidden.update({f"TXT{n}Q", f"SNP{n}Q"})
    spec, _ = spec_for({LEG: Route(legislation_body(hits, AS_OF))}, unresolved=policy)
    out = spec.lexapi_search_legislation("q", as_of=AS_OF)
    for token in forbidden:
        assert token not in out
    for n, kind in enumerate(spec_list):
        if kind == "resolved":
            assert f"TXT{n}Q" in out
