"""Behaviour of the LexAPI tool spec, offline, against synthetic fixtures."""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import pickle
import subprocess
import sys
import textwrap
import traceback
from collections.abc import Callable
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

from llama_index.tools.lexapi import (
    LexAPICallReport,
    LexAPIToolError,
    LexAPIToolEvent,
    LexAPIToolSpec,
)
from llama_index.tools.lexapi._passages import cut_text

LEG = "/legislation/semantic"
DOC = "/documentContent"
ASOF = "/legislation/as-of"
AS_OF = "2099-06-01"
SECRET = "lex_secret_key_must_not_leak_42"


def spec_for(routes: dict[str, Route], **kwargs: Any) -> tuple[LexAPIToolSpec, list[Any]]:
    client, recorder = make_client(routes)
    return LexAPIToolSpec(client, **kwargs), recorder.requests


def tools_by_name(spec: LexAPIToolSpec) -> dict[str, FunctionTool]:
    return {t.metadata.name or "": t for t in spec.to_tool_list()}


def numbered(blocks: list[str]) -> str:
    return "\n\n".join(f"[{n}] {b}" for n, b in enumerate(blocks, 1))


class _Collector(BaseEventHandler):
    events: list[BaseEvent] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "Collector"

    def handle(self, event: BaseEvent, **kwargs: Any) -> None:
        if isinstance(event, LexAPIToolEvent):
            self.events.append(event)


class collect:
    """Context manager collecting LexAPIToolEvent instances."""

    def __enter__(self) -> list[LexAPIToolEvent]:
        self.collector = _Collector()
        get_dispatcher().add_event_handler(self.collector)
        return self.collector.events  # type: ignore[return-value]

    def __exit__(self, *exc: object) -> None:
        get_dispatcher().event_handlers.remove(self.collector)


# ── shape: names, descriptions, signatures, schemas ────────────────────

POLICY_SENTENCES = {
    "withhold": (
        "an article whose text LexAPI did not establish for that date comes back as a notice "
        "without text."
    ),
    "drop": (
        "an article whose text LexAPI did not establish for that date is left out, and the "
        "output says how many were left out."
    ),
    "keep_labelled": (
        "for an article whose text LexAPI did not establish for that date you get its latest "
        "text, under a label saying it is not established for that date."
    ),
}


def search_text(policy: str) -> str:
    return (
        "Search EU legislation (EUR-Lex) by meaning. Returns matching articles, each followed "
        "by a Source line with its CELEX citation. With as_of (YYYY-MM-DD) it asks for the "
        "text for that date; "
        + POLICY_SENTENCES[policy]
        + " Consolidated texts have no legal force; the Source line names the act to cite "
        "when LexAPI knows it."
    )


READ_LEAD = (
    "Read an EU document by its CELEX number: one article (article argument), or its "
    "articles in order. Each passage is followed by a Source line with its CELEX citation. "
    "Consolidated texts (CELEX starting with 0) have no legal force: cite the act as "
    "published in the Official Journal."
)
AS_OF_TEXT = (
    "The version of an EU act that LexAPI lists for a date (as_of, YYYY-MM-DD), and "
    "optionally one provision of it, followed by a Source line with its CELEX citation. "
    "When LexAPI did not establish the text for that date, the answer is a notice without "
    "text. It does not say whether a provision applied on that date. Consolidated texts have "
    "no legal force; the Source line names the act to cite when LexAPI knows it."
)
SIGNATURES = {
    "lexapi_search_legislation": "(query: str, as_of: str | None = None) -> str",
    "lexapi_read_document": (
        "(celex: str, article: str | None = None, language: str | None = None) -> str"
    ),
    "lexapi_legislation_as_of": (
        "(celex: str, as_of: str, provision: str | None = None, language: str | None = None) -> str"
    ),
}
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
LANGUAGE_PATTERN = r"^[a-z]{2}$"


def read_text(max_chars: int | None) -> str:
    if max_chars is None:
        return READ_LEAD
    return (
        f"{READ_LEAD} The output is limited by a character budget of {max_chars} characters "
        "of text; the articles not shown are listed by reference."
    )


def test_spec_derives_from_base_tool_spec() -> None:
    assert BaseToolSpec in LexAPIToolSpec.__mro__


def test_spec_functions_are_pinned() -> None:
    assert LexAPIToolSpec.spec_functions == [
        ("lexapi_search_legislation", "alexapi_search_legislation"),
        ("lexapi_read_document", "alexapi_read_document"),
        ("lexapi_legislation_as_of", "alexapi_legislation_as_of"),
    ]


@pytest.mark.parametrize("policy", ["withhold", "drop", "keep_labelled"])
@pytest.mark.parametrize("max_chars", [20_000, 500, None])
def test_descriptions_are_pinned(policy: Any, max_chars: int | None) -> None:
    spec, _ = spec_for({}, unresolved=policy, max_chars=max_chars)
    tools = tools_by_name(spec)
    assert list(tools) == list(SIGNATURES)
    texts = {
        "lexapi_search_legislation": search_text(policy),
        "lexapi_read_document": read_text(max_chars),
        "lexapi_legislation_as_of": AS_OF_TEXT,
    }
    for name, signature in SIGNATURES.items():
        assert str(inspect.signature(getattr(spec, name))) == signature
        assert str(inspect.signature(getattr(spec, "a" + name))) == signature
        assert tools[name].metadata.description == f"{name}{signature}\n{texts[name]}"


def test_default_read_description_names_the_budget() -> None:
    spec, _ = spec_for({})
    assert tools_by_name(spec)["lexapi_read_document"].metadata.description.endswith(
        "The output is limited by a character budget of 20000 characters of text; "
        "the articles not shown are listed by reference."
    )


def test_argument_schemas_are_pinned() -> None:
    spec, _ = spec_for({})
    params = {n: t.metadata.get_parameters_dict() for n, t in tools_by_name(spec).items()}
    search = params["lexapi_search_legislation"]
    assert search["required"] == ["query"]
    assert search["properties"]["query"]["minLength"] == 1
    assert search["properties"]["as_of"]["anyOf"][0] == {
        "pattern": DATE_PATTERN,
        "type": "string",
    }
    read = params["lexapi_read_document"]
    assert read["required"] == ["celex"]
    assert read["properties"]["celex"]["minLength"] == 1
    assert read["properties"]["language"]["anyOf"][0] == {
        "pattern": LANGUAGE_PATTERN,
        "type": "string",
    }
    as_of = params["lexapi_legislation_as_of"]
    assert as_of["required"] == ["celex", "as_of"]
    assert as_of["properties"]["celex"]["minLength"] == 1
    assert as_of["properties"]["as_of"]["pattern"] == DATE_PATTERN
    assert as_of["properties"]["language"]["anyOf"][0]["pattern"] == LANGUAGE_PATTERN
    for name, schema in params.items():
        signature = inspect.signature(getattr(spec, name))
        assert list(schema["properties"]) == list(signature.parameters)


def test_every_tool_has_a_real_async_variant() -> None:
    spec, _ = spec_for({})
    for name in SIGNATURES:
        assert inspect.iscoroutinefunction(getattr(spec, "a" + name))
        assert not inspect.iscoroutinefunction(getattr(spec, name))


@pytest.mark.parametrize("policy", ["withhold", "drop", "keep_labelled"])
def test_descriptions_make_no_compliance_or_advice_claims(policy: Any) -> None:
    spec, _ = spec_for({}, unresolved=policy)
    text = " ".join(
        (t.metadata.description or "") + json.dumps(t.metadata.get_parameters_dict())
        for t in spec.to_tool_list()
    ).lower()
    for word in ("complian", "legal advice", "in force", "guarantee", "accurate"):
        assert word not in text


# ── search ──────────────────────────────────────────────────────────────


def expected_blocks(fixture: str, as_of: str | None, policy: Any = "withhold") -> list[str]:
    result = passages.from_legislation_search(
        SemanticLegislationResponse(load_fixture(fixture)), unresolved=policy, as_of=as_of
    )
    return [passages.format_passage(p) for p in result.passages]


def test_search_request_and_output() -> None:
    spec, requests = spec_for({LEG: fixture_route("legislation.mixed.json")}, similarity_top_k=3)
    out = spec.lexapi_search_legislation("erasure", as_of=AS_OF)
    assert json.loads(requests[0].content) == {
        "query": "erasure",
        "limit": 3,
        "include": ["text"],
        "asOf": AS_OF,
    }
    assert out == numbered(expected_blocks("legislation.mixed.json", AS_OF))
    assert out.startswith("[1] [SYNTHETIC FIXTURE")
    assert "\n\n[3] [No text: CELEX 32099L0003 art_12" in out
    assert "UNRESOLVEDTEXTTOKEN" not in out
    assert "UNRESOLVEDSNIPPETTOKEN" not in out


def test_search_without_a_date() -> None:
    spec, requests = spec_for({LEG: fixture_route("legislation.no-date.json")})
    out = spec.lexapi_search_legislation("erasure")
    assert "asOf" not in json.loads(requests[0].content)
    assert out == numbered(expected_blocks("legislation.no-date.json", None))


def test_search_checks_the_date_on_every_hit() -> None:
    body = load_fixture("legislation.as-published.json")
    body["results"][0]["asOf"] = "2099-01-01"
    spec, _ = spec_for({LEG: Route(body)})
    out = spec.lexapi_search_legislation("q", as_of=AS_OF)
    assert "as published article five" not in out
    assert out.startswith("[1] [No text: CELEX 32099R0002 art_5")


def test_search_drop_note_follows_the_passages() -> None:
    spec, _ = spec_for({LEG: fixture_route("legislation.mixed.json")}, unresolved="drop")
    out = spec.lexapi_search_legislation("q", as_of=AS_OF)
    assert out == (
        numbered(expected_blocks("legislation.mixed.json", AS_OF, "drop"))
        + "\n\n[1 hit(s) left out: LexAPI did not establish their text for 2099-06-01.]"
    )
    spec, _ = spec_for({LEG: fixture_route("legislation.unresolved.json")}, unresolved="drop")
    assert spec.lexapi_search_legislation("q", as_of=AS_OF) == (
        "LexAPI returned no matching articles.\n\n"
        "[1 hit(s) left out: LexAPI did not establish their text for 2099-06-01.]"
    )


def test_search_drop_note_without_a_known_date() -> None:
    from llama_index.tools.lexapi.base import _render

    result = passages.PassageResult(passages=(), dropped=2)
    text, _ = _render(result, requested=5, empty="LexAPI returned no matching articles.")
    assert text.endswith(
        "[2 hit(s) left out: LexAPI did not establish their text for the date asked.]"
    )


def test_search_empty_and_hint_order() -> None:
    body = load_fixture("legislation.empty.json")
    spec, _ = spec_for({LEG: Route(body)})
    assert spec.lexapi_search_legislation("q") == "LexAPI returned no matching articles."
    body = load_fixture("legislation.unresolved.json")
    body["hint"] = "Few results; try other words."
    spec, _ = spec_for({LEG: Route(body)}, unresolved="drop")
    assert spec.lexapi_search_legislation("q", as_of=AS_OF) == (
        "LexAPI returned no matching articles.\n\n"
        "[1 hit(s) left out: LexAPI did not establish their text for 2099-06-01.]\n\n"
        "Hint from LexAPI: Few results; try other words."
    )


def test_search_output_is_de_duplicated_by_rank() -> None:
    body = load_fixture("legislation.unresolved.json")
    first = dict(body["results"][0])
    second = dict(first, asOfResolved=True, text="RESOLVED-TWIN")
    second.pop("asOfReason")
    body["results"] = [first, second]
    spec, _ = spec_for({LEG: Route(body)})
    out = spec.lexapi_search_legislation("q", as_of=AS_OF)
    assert out.count("Source:") == 1
    assert out.startswith("[1] RESOLVED-TWIN")


# ── dates validated before any request ─────────────────────────────────

BAD_DATES = ["2099-02-30", "2099-6-1", "01.06.2099", "", "2099-06-01T00:00", "lex_abcdefgh"]


@pytest.mark.parametrize("bad", BAD_DATES)
def test_bad_dates_are_refused_before_any_request(bad: str) -> None:
    routes = {
        LEG: fixture_route("legislation.mixed.json"),
        ASOF: fixture_route("as-of.resolved.json"),
    }
    spec, requests = spec_for(routes)
    calls: list[Callable[[], str]] = [
        lambda: spec.lexapi_search_legislation("q", as_of=bad),
        lambda: spec.lexapi_legislation_as_of("32099R0001", bad),
    ]
    for call in calls:
        with pytest.raises(LexAPIToolError) as info:
            call()
        shown = bad.replace("lex_abcdefgh", "lex_***")
        assert str(info.value) == (
            f"Invalid argument: as_of must be a date in the form YYYY-MM-DD; got '{shown}'"
        )
        assert info.value.code == "INVALID_ARGUMENT"
    assert requests == []


@pytest.mark.parametrize("bad", ["2099-02-30", "x"])
async def test_bad_dates_are_refused_async(bad: str) -> None:
    aclient, recorder = make_async_client({})
    spec = LexAPIToolSpec(async_client=aclient)
    with pytest.raises(LexAPIToolError, match="Invalid argument: as_of"):
        await spec.alexapi_search_legislation("q", as_of=bad)
    with pytest.raises(LexAPIToolError, match="Invalid argument: as_of"):
        await spec.alexapi_legislation_as_of("32099R0001", bad)
    assert recorder.requests == []


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
    assert out == numbered([passages.format_passage(p) for p in result.passages])


def test_read_whole_document() -> None:
    spec, requests = spec_for({DOC: fixture_route("document.read.json")}, language="fr")
    out = spec.lexapi_read_document("32099R0001")
    assert json.loads(requests[0].content) == {
        "celexNumber": "32099R0001",
        "language": "fr",
        "include": ["metadata", "articles", "fullText"],
    }
    assert out.startswith("[1] ")
    assert "\n\n[2] " in out
    assert out.count("Source: CELEX 32099R0001") == 2


def test_read_document_without_text() -> None:
    body = load_fixture("document.read.json")
    body["document"]["content"] = None
    spec, _ = spec_for({DOC: Route(body)})
    assert (
        spec.lexapi_read_document("32099R0001", article="art_9")
        == "LexAPI returned no text for article art_9 of CELEX 32099R0001."
    )
    assert (
        spec.lexapi_read_document("32099R0001") == "LexAPI returned no text for CELEX 32099R0001."
    )


def long_document(lengths: list[int]) -> dict[str, Any]:
    body: dict[str, Any] = load_fixture("document.read.json")
    body["document"]["content"]["articles"] = [
        {"id": f"art_{n}", "number": f"Article {n}", "content": ("w" * 9 + " ") * (size // 10)}
        for n, size in enumerate(lengths, 1)
    ]
    return body


def _shown_text_total(out: str) -> int:
    """Characters of article text in a read output (Source lines, numbers
    and notes excluded)."""
    total = 0
    for block in out.split("\n\n"):
        if not block.startswith("["):
            continue
        if block.startswith("[Not shown"):
            continue
        body = block.split("] ", 1)[1].split("\nSource: ", 1)[0]
        body = body.split("\n[Text cut by ", 1)[0]
        total += len(body)
    return total


def test_read_budget_bounds_the_sum_of_the_texts() -> None:
    spec, _ = spec_for({DOC: Route(long_document([300, 300, 300, 300, 300]))}, max_chars=1000)
    with collect() as events:
        out = spec.lexapi_read_document("32099R0001")
    assert _shown_text_total(out) <= 1000
    # 3 texts of 299 characters fit (897), the fourth would not.
    assert out.count("Source: CELEX") == 3
    assert out.endswith(
        "\n\n[Not shown: 2 more article(s) of CELEX 32099R0001: art_4, art_5. "
        "Read one with the article argument.]"
    )
    (event,) = events
    assert (event.report.returned, event.report.not_shown, event.report.truncated) == (3, 2, 0)
    assert event.report.requested is None


def test_read_budget_cuts_the_first_text_with_the_note() -> None:
    spec, _ = spec_for({DOC: Route(long_document([5000, 100]))}, max_chars=1000)
    with collect() as events:
        out = spec.lexapi_read_document("32099R0001")
    assert _shown_text_total(out) <= 1000
    assert "\n[Text cut by llama-index-tools-lexapi: 999 of 5000 characters shown. " in out
    assert "The full text is at the source.]\nSource: CELEX 32099R0001" in out
    assert out.endswith(
        "[Not shown: 1 more article(s) of CELEX 32099R0001: art_2. "
        "Read one with the article argument.]"
    )
    (event,) = events
    assert (event.report.truncated, event.report.not_shown) == (1, 1)


def test_read_budget_none_means_no_bound() -> None:
    spec, _ = spec_for({DOC: Route(long_document([30_000, 30_000]))}, max_chars=None)
    out = spec.lexapi_read_document("32099R0001")
    assert "Not shown" not in out and "Text cut" not in out
    assert _shown_text_total(out) > 50_000


def test_read_default_budget_is_20000() -> None:
    spec, _ = spec_for({DOC: Route(long_document([1000] * 30))})
    out = spec.lexapi_read_document("32099R0001")
    assert "[Not shown: " in out
    assert _shown_text_total(out) <= 20_000
    assert "max_chars=20000" in repr(spec)


@pytest.mark.parametrize("bad", [199, 0, True, 1.5])
def test_read_max_chars_is_validated(bad: Any) -> None:
    client, _ = make_client({})
    with pytest.raises(ValueError, match="max_chars"):
        LexAPIToolSpec(client, max_chars=bad)


# ── the cut rule (tools copy) ───────────────────────────────────────────


def test_cut_rule_example() -> None:
    text, cut = cut_text("aaaa bbbbb cccc", 10)
    assert cut
    assert text == (
        "aaaa bbbbb\n[Text cut by llama-index-tools-lexapi: 10 of 15 characters shown. "
        "The full text is at the source.]"
    )


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("a" * 150 + "\n\n" + "b" * 100, "a" * 150),
        ("a" * 150 + "\n" + "b" * 100, "a" * 150),
        ("a" * 150 + " " + "b" * 100, "a" * 150),
        ("a" * 250, "a" * 200),
        ("   " + "a" * 250, "a" * 200),
        ("a" * 200 + " " + "b", "a" * 200),
    ],
)
def test_cut_rule_bounds(text: str, kept: str) -> None:
    out, cut = cut_text(text, 200)
    assert cut
    assert out.split("\n[Text cut by ")[0] == kept
    assert len(kept) <= 200
    assert f"{len(kept)} of {len(text)} characters shown" in out


def test_cut_rule_returns_a_fitting_text_unchanged() -> None:
    assert cut_text("\n  short", 200) == ("\n  short", False)


def test_cut_rule_ignores_a_break_in_the_first_half() -> None:
    words = " ".join(f"word{n}" for n in range(2000))
    text = "Heading\n\n" + words
    out, cut = cut_text(text, 4000)
    kept = out.split("\n[Text cut by ", 1)[0]
    assert cut
    assert 3990 <= len(kept) <= 4000
    assert text.startswith(kept) and text[len(kept)] == " "


# ── as of a date ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("fixture", "provision", "starts"),
    [
        ("as-of.resolved.json", "art_17", "[SYNTHETIC FIXTURE"),
        ("as-of.unresolved.json", "art_17", "[No text: CELEX 32099R0001 art_17 is not established"),
        ("as-of.version-only.json", None, "[Version 02099R0001-20990101 (consolidated) is the"),
    ],
)
def test_legislation_as_of(fixture: str, provision: str | None, starts: str) -> None:
    spec, requests = spec_for({ASOF: fixture_route(fixture)})
    out = spec.lexapi_legislation_as_of("32099R0001", AS_OF, provision=provision)
    sent = {"document": "32099R0001", "asOf": AS_OF}
    if provision:
        sent["provision"] = provision
    assert json.loads(requests[0].content) == sent
    result = passages.from_legislation_as_of(
        LegislationAsOfResponse(load_fixture(fixture)), provision=provision
    )
    assert out == passages.format_passage(result.passages[0])
    assert out.startswith(starts)


def test_legislation_as_of_names_the_act_to_cite() -> None:
    spec, _ = spec_for({ASOF: fixture_route("as-of.resolved.json")})
    out = spec.lexapi_legislation_as_of("32099R0001", AS_OF, provision="art_17")
    assert (
        "Source: CELEX 32099R0001, art_17, consolidated text 02099R0001-20990101 (no legal force)"
        in out
    )
    assert "version for 2099-06-01: 02099R0001-20990101" in out


@pytest.mark.parametrize("policy", ["withhold", "drop", "keep_labelled"])
def test_legislation_as_of_always_withholds(policy: Any) -> None:
    spec, _ = spec_for({ASOF: fixture_route("as-of.unresolved.json")}, unresolved=policy)
    out = spec.lexapi_legislation_as_of("32099R0001", AS_OF, provision="art_17")
    assert out.startswith("[No text: CELEX 32099R0001 art_17 is not established")
    assert "amendment_not_consolidated" in out


def test_legislation_as_of_without_a_document_is_unreadable() -> None:
    body = load_fixture("as-of.unresolved.json")
    body["document"] = None
    spec, _ = spec_for({ASOF: Route(body)})
    with pytest.raises(LexAPIToolError) as info:
        spec.lexapi_legislation_as_of("32099R0001", AS_OF)
    assert str(info.value) == "LexAPI error INVALID_RESPONSE: The answer could not be read."


# ── errors ──────────────────────────────────────────────────────────────

ERRORS = [
    ("error.401.json", 401, "UNAUTHORIZED", {}),
    ("error.402.json", 402, "CREDITS_EXHAUSTED", {}),
    ("error.403-past-due.json", 403, "PAYMENT_PAST_DUE", {}),
    ("error.429.json", 429, "RATE_LIMITED", {"Retry-After": "30"}),
]
EXACT = {
    401: "LexAPI error UNAUTHORIZED (HTTP 401): Invalid API key.",
    402: "LexAPI error CREDITS_EXHAUSTED (HTTP 402): Monthly credit pool exhausted. "
    "Credits reset at 2099-07-01T00:00:00.000Z.",
    403: "LexAPI error PAYMENT_PAST_DUE (HTTP 403): Payment past due. "
    "Billing: https://lex-api.com/dashboard#billing",
    429: "LexAPI error RATE_LIMITED (HTTP 429): Rate limit exceeded. Retry after 30 s.",
}

CALLS = [
    ("lexapi_search_legislation", {"query": "q"}),
    ("lexapi_read_document", {"celex": "32099R0001"}),
    ("lexapi_legislation_as_of", {"celex": "32099R0001", "as_of": AS_OF}),
]


def error_routes(fixture: str, status: int, headers: dict[str, str]) -> dict[str, Route]:
    return {path: fixture_route(fixture, status, headers) for path in (LEG, DOC, ASOF)}


def assert_detached(error: BaseException) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.parametrize(("fixture", "status", "code", "headers"), ERRORS)
def test_errors_raise_with_exact_text(
    fixture: str, status: int, code: str, headers: dict[str, str]
) -> None:
    spec, _ = spec_for(error_routes(fixture, status, headers))
    for name, args in CALLS:
        with pytest.raises(LexAPIToolError) as info:
            getattr(spec, name)(**args)
        assert str(info.value) == EXACT[status]
        assert (info.value.code, info.value.status) == (code, status)
        assert_detached(info.value)


@pytest.mark.parametrize(("fixture", "status", "code", "headers"), ERRORS)
async def test_errors_raise_async(
    fixture: str, status: int, code: str, headers: dict[str, str]
) -> None:
    aclient, _ = make_async_client(error_routes(fixture, status, headers))
    spec = LexAPIToolSpec(async_client=aclient)
    for name, args in CALLS:
        with pytest.raises(LexAPIToolError) as info:
            await getattr(spec, "a" + name)(**args)
        assert str(info.value) == EXACT[status]
        assert_detached(info.value)


@pytest.mark.parametrize(("fixture", "status", "code", "headers"), ERRORS)
async def test_errors_reach_the_agent_as_error_output(
    fixture: str, status: int, code: str, headers: dict[str, str]
) -> None:
    spec, _ = spec_for(error_routes(fixture, status, headers))
    tools = tools_by_name(spec)
    for name, args in CALLS:
        for output in (call_tool(tools[name], args), await acall_tool(tools[name], args)):
            assert output.is_error is True
            assert EXACT[status] in output.content
            assert "lex_offline_test_key" not in output.content


@pytest.mark.parametrize(
    ("message", "rendered"),
    [("Stop.", "Stop."), ("Why?", "Why?"), ("Wow!", "Wow!"), ("Plain", "Plain.")],
)
def test_error_period_only_when_missing(message: str, rendered: str) -> None:
    error = lexapi.LexAPIError(message, code="X", status=400)
    assert str(LexAPIToolError.from_sdk(error)) == f"LexAPI error X (HTTP 400): {rendered}"


def test_retry_after_for_any_error_carrying_a_wait() -> None:
    error = lexapi.ServiceUnavailableError(
        "Busy", code="SERVICE_UNAVAILABLE", status=503, retry_after=2.5
    )
    assert str(LexAPIToolError.from_sdk(error)) == (
        "LexAPI error SERVICE_UNAVAILABLE (HTTP 503): Busy. Retry after 2.5 s."
    )


def test_key_shapes_are_masked() -> None:
    error = lexapi.LexAPIError("bad lex_a-b_C", code="X", status=401)
    assert str(LexAPIToolError.from_sdk(error)) == "LexAPI error X (HTTP 401): bad lex_***."


# ── an unreadable answer never leaks its body ──────────────────────────

LEAK = "UNRESOLVEDTEXTTOKEN"


def truncated_body() -> str:
    body = json.dumps(
        {"results": [{"text": f"{LEAK} latest indexed article twelve", "asOfResolved": False}]}
    )
    cut = body[: body.index(LEAK) + 60]
    assert LEAK in cut[:200]
    return cut


def leak_routes() -> dict[str, Route]:
    return {path: Route(truncated_body()) for path in (LEG, DOC, ASOF)}


def chain(error: BaseException) -> list[str]:
    out = []
    seen: BaseException | None = error
    while seen is not None:
        out.append(repr(seen))
        out.append(str(seen))
        out.extend(map(str, seen.args))
        seen = seen.__cause__ or seen.__context__
    out.extend(traceback.format_exception(error))
    return out


def test_the_sdk_keeps_the_body_out_of_its_message() -> None:
    client, _ = make_client(leak_routes())
    with pytest.raises(lexapi.InvalidResponseError) as info:
        client.semantic_legislation_search("q")
    assert LEAK not in str(info.value)
    assert LEAK in (info.value.body_excerpt or "")


def test_unreadable_answer_text_is_fixed_and_detached() -> None:
    spec, _ = spec_for(leak_routes())
    with collect() as events:
        for name, args in CALLS:
            with pytest.raises(LexAPIToolError) as info:
                getattr(spec, name)(**args)
            assert str(info.value) == (
                "LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read."
            )
            assert_detached(info.value)
            assert not any(LEAK in part for part in chain(info.value))
    assert events == []


async def test_unreadable_answer_async_and_through_the_agent() -> None:
    aclient, _ = make_async_client(leak_routes())
    spec = LexAPIToolSpec(async_client=aclient)
    tools = tools_by_name(spec)
    with collect() as events:
        for name, args in CALLS:
            with pytest.raises(LexAPIToolError) as info:
                await getattr(spec, "a" + name)(**args)
            assert_detached(info.value)
            assert not any(LEAK in part for part in chain(info.value))
            for output in (call_tool(tools[name], args), await acall_tool(tools[name], args)):
                assert output.is_error
                assert LEAK not in output.content
                assert LEAK not in str(output.raw_output)
                # ToolOutput.exception exists from llama-index-core 0.14.
                exception = getattr(output, "exception", None)
                if exception is not None:
                    assert not any(LEAK in part for part in chain(exception))
    assert events == []


def test_any_sdk_error_on_a_2xx_status_gets_the_fixed_text() -> None:
    error = lexapi.LexAPIError(f"weird {LEAK}", code="SOMETHING", status=204)
    assert str(LexAPIToolError.from_sdk(error)) == (
        "LexAPI error INVALID_RESPONSE (HTTP 204): The answer could not be read."
    )
    no_status = lexapi.InvalidResponseError(f"x {LEAK}")
    assert str(LexAPIToolError.from_sdk(no_status)) == (
        "LexAPI error INVALID_RESPONSE: The answer could not be read."
    )


def test_retrievers_side_is_unchanged_the_sdk_error_propagates() -> None:
    # The tools wrap; a raw SDK call still raises the SDK error.
    client, _ = make_client(leak_routes())
    with pytest.raises(lexapi.InvalidResponseError):
        client.get_document("32099R0001")


# ── sync == async ───────────────────────────────────────────────────────

ROUTES = {
    LEG: fixture_route("legislation.mixed.json"),
    DOC: fixture_route("document.read.json"),
    ASOF: fixture_route("as-of.resolved.json"),
}
OK_CALLS = [
    ("lexapi_search_legislation", {"query": "q", "as_of": AS_OF}),
    ("lexapi_read_document", {"celex": "32099R0001", "article": "art_17"}),
    ("lexapi_legislation_as_of", {"celex": "32099R0001", "as_of": AS_OF, "provision": "art_17"}),
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


def test_sync_with_only_an_async_client_is_a_value_error() -> None:
    aclient, _ = make_async_client({})
    spec = LexAPIToolSpec(async_client=aclient)
    with pytest.raises(ValueError) as info:
        spec.lexapi_search_legislation("q")
    assert str(info.value) == (
        "LexAPIToolSpec was given only an async_client; use the async methods or pass "
        "client= as well"
    )


# ── report ──────────────────────────────────────────────────────────────

REPORT_FIELDS = [
    "requested",
    "returned",
    "as_of",
    "withheld",
    "kept_labelled",
    "dropped",
    "uncitable",
    "filtered_out",
    "returned_withheld",
    "returned_labelled",
    "withheld_not_emitted",
    "duplicates",
    "truncated",
    "not_shown",
    "units_charged",
    "credits_remaining",
    "hint",
]


def test_report_fields_are_pinned() -> None:
    assert [f.name for f in dataclasses.fields(LexAPICallReport)] == REPORT_FIELDS
    assert LexAPICallReport.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_each_call_dispatches_its_report() -> None:
    spec, _ = spec_for(ROUTES, similarity_top_k=7)
    with collect() as events:
        for name, args in OK_CALLS:
            getattr(spec, name)(**args)
    assert [(e.tool, e.report.requested, e.report.units_charged) for e in events] == [
        ("lexapi_search_legislation", 7, 9),
        ("lexapi_read_document", None, 1),
        ("lexapi_legislation_as_of", None, 4),
    ]
    assert events[0].report == LexAPICallReport(
        requested=7,
        returned=3,
        as_of=AS_OF,
        withheld=1,
        kept_labelled=0,
        dropped=0,
        uncitable=0,
        filtered_out=0,
        returned_withheld=1,
        returned_labelled=0,
        withheld_not_emitted=0,
        duplicates=0,
        truncated=0,
        not_shown=0,
        units_charged=9,
        credits_remaining=991,
        hint=None,
    )


def test_report_counts_labelled_passages() -> None:
    spec, _ = spec_for({LEG: fixture_route("legislation.mixed.json")}, unresolved="keep_labelled")
    with collect() as events:
        spec.lexapi_search_legislation("q", as_of=AS_OF)
    assert (events[0].report.kept_labelled, events[0].report.returned_labelled) == (1, 1)


# ── clients, lifecycle, secrets ─────────────────────────────────────────


def test_validation() -> None:
    client, _ = make_client({})
    with pytest.raises(ValueError, match="similarity_top_k"):
        LexAPIToolSpec(client, similarity_top_k=0)
    with pytest.raises(ValueError, match="similarity_top_k"):
        LexAPIToolSpec(client, similarity_top_k=51)
    with pytest.raises(ValueError, match="unresolved"):
        LexAPIToolSpec(client, unresolved="latest_labelled")  # type: ignore[arg-type]
    with pytest.raises(ValueError) as info:
        LexAPIToolSpec(client, api_key="lex_x")
    assert str(info.value) == "Pass either client/async_client or api_key/base_url, not both."


def test_unknown_constructor_arguments_raise() -> None:
    client, _ = make_client({})
    with pytest.raises(TypeError):
        LexAPIToolSpec(client, k=3)  # type: ignore[call-arg]


def test_key_is_not_in_repr_and_owned_clients_cannot_reach_the_network() -> None:
    spec = LexAPIToolSpec(api_key=SECRET, base_url=LOOPBACK)
    for text in (repr(spec), str(spec), repr(vars(spec))):
        assert SECRET not in text
    sync = spec._clients._sync
    assert sync is not None
    assert sync._http.headers["user-agent"].endswith("llama-index-tools-lexapi/0.1.0")
    with pytest.raises(RuntimeError, match="Offline tests blocked"):
        spec.lexapi_search_legislation("q")
    spec.close()


def test_default_base_url_cannot_be_reached() -> None:
    spec = LexAPIToolSpec(api_key=SECRET)
    with pytest.raises(RuntimeError, match="Offline tests blocked"):
        spec.lexapi_legislation_as_of("32099R0001", AS_OF)


def test_context_manager_closes_and_marks_closed() -> None:
    with LexAPIToolSpec(api_key=SECRET, base_url=LOOPBACK) as spec:
        sync = spec._clients._sync
        assert spec._clients._async is None
    assert sync is not None and sync._http.is_closed
    for name, args in CALLS:
        with pytest.raises(LexAPIToolError) as info:
            getattr(spec, name)(**args)
        assert str(info.value) == "LexAPIToolSpec is closed; create a new one."
        assert_detached(info.value)


async def test_async_context_manager_closes_both_owned_clients() -> None:
    async with LexAPIToolSpec(api_key=SECRET, base_url=LOOPBACK) as spec:
        aclient = spec._clients.async_
        assert aclient is not None
    assert aclient._http.is_closed
    for name, args in CALLS:
        with pytest.raises(
            LexAPIToolError, match=r"^LexAPIToolSpec is closed; create a new one\.$"
        ):
            await getattr(spec, "a" + name)(**args)


async def test_passed_clients_are_never_closed() -> None:
    client, _ = make_client({LEG: fixture_route("legislation.empty.json")})
    aclient, _ = make_async_client({})
    spec = LexAPIToolSpec(client, aclient)
    spec.close()
    await spec.aclose()
    assert not client._http.is_closed and not aclient._http.is_closed


def test_tool_spec_cannot_be_copied_or_pickled() -> None:
    spec, _ = spec_for({})
    for attempt in (
        lambda: pickle.dumps(spec),
        lambda: copy.copy(spec),
        lambda: copy.deepcopy(spec),
    ):
        with pytest.raises(TypeError) as info:
            attempt()
        assert str(info.value) == (
            "LexAPIToolSpec holds an open client and cannot be copied or pickled."
        )


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


# ── wrong-shape 200s: never an empty answer ────────────────────────────

WRONG_SHAPES = {
    "results not a list": {"success": True, "results": {"a": 1}},
    "success false": {
        "success": False,
        "error": {"code": "UPSTREAM_ERROR", "message": "Upstream failed"},
    },
    "empty body": "",
}
FIXED = "LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read."


@pytest.mark.parametrize("shape", ["success false", "empty body"])
def test_wrong_shape_200_gives_the_fixed_text_on_every_tool(shape: str) -> None:
    routes = {path: Route(WRONG_SHAPES[shape]) for path in (LEG, DOC, ASOF)}
    spec, _ = spec_for(routes)
    with collect() as events:
        for name, args in CALLS:
            with pytest.raises(LexAPIToolError) as info:
                getattr(spec, name)(**args)
            assert str(info.value) == FIXED
            assert_detached(info.value)
    assert events == []


@pytest.mark.parametrize("shape", list(WRONG_SHAPES))
async def test_wrong_shape_200_search_sync_and_async(shape: str) -> None:
    spec, _ = spec_for({LEG: Route(WRONG_SHAPES[shape])})
    aclient, _ = make_async_client({LEG: Route(WRONG_SHAPES[shape])})
    aspec = LexAPIToolSpec(async_client=aclient)
    with pytest.raises(LexAPIToolError) as info:
        spec.lexapi_search_legislation("q")
    assert str(info.value) == FIXED
    with pytest.raises(LexAPIToolError) as ainfo:
        await aspec.alexapi_search_legislation("q")
    assert str(ainfo.value) == FIXED


# ── malformed 200s on read and as-of: an error, never "no text" ────────

NO_STATUS = "LexAPI error INVALID_RESPONSE: The answer could not be read."
WITH_STATUS = "LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not be read."

READ_BAD = {
    "empty object": {},
    "success only": {"success": True},
    "document null": {"success": True, "document": None},
    "document a string": {"success": True, "document": "x"},
    "content a string": {"success": True, "document": {"celex": "32099R0001", "content": "x"}},
    "articles not a list": {
        "success": True,
        "document": {"celex": "32099R0001", "content": {"articles": "x"}},
    },
    "article not an object": {
        "success": True,
        "document": {"celex": "32099R0001", "content": {"articles": ["x"]}},
    },
}
AS_OF_BAD = {
    "empty object": {},
    "success only": {"success": True},
    "document null": {"success": True, "document": None, "asOfResolved": False},
    "document empty": {"success": True, "document": "", "asOfResolved": False},
    "document an object": {"success": True, "document": {}, "asOfResolved": False},
    "resolved missing": {"success": True, "document": "32099R0001"},
    "resolved a string": {"success": True, "document": "32099R0001", "asOfResolved": "true"},
}


async def _all_paths(
    spec: LexAPIToolSpec, aspec: LexAPIToolSpec, name: str, args: dict[str, Any]
) -> list[str]:
    out = []
    with pytest.raises(LexAPIToolError) as info:
        getattr(spec, name)(**args)
    assert_detached(info.value)
    out.append(str(info.value))
    with pytest.raises(LexAPIToolError) as ainfo:
        await getattr(aspec, "a" + name)(**args)
    assert_detached(ainfo.value)
    out.append(str(ainfo.value))
    tool = tools_by_name(spec)[name]
    for output in (call_tool(tool, args), await acall_tool(tool, args)):
        assert output.is_error
        out.append(output.content)
    return out


@pytest.mark.parametrize("shape", list(READ_BAD))
async def test_malformed_read_answers_are_unreadable(shape: str) -> None:
    routes = {DOC: Route(READ_BAD[shape])}
    spec, _ = spec_for(routes)
    aclient, _ = make_async_client(routes)
    aspec = LexAPIToolSpec(async_client=aclient)
    with collect() as events:
        texts = await _all_paths(spec, aspec, "lexapi_read_document", {"celex": "32099R0001"})
    assert all(NO_STATUS in t for t in texts)
    assert events == []


@pytest.mark.parametrize("shape", list(AS_OF_BAD))
async def test_malformed_as_of_answers_are_unreadable(shape: str) -> None:
    routes = {ASOF: Route(AS_OF_BAD[shape])}
    spec, _ = spec_for(routes)
    aclient, _ = make_async_client(routes)
    aspec = LexAPIToolSpec(async_client=aclient)
    args = {"celex": "32099R0001", "as_of": AS_OF}
    with collect() as events:
        texts = await _all_paths(spec, aspec, "lexapi_legislation_as_of", args)
    assert all(NO_STATUS in t for t in texts)
    assert events == []


@pytest.mark.parametrize("shape", ["empty object", "success only", "document null"])
async def test_malformed_search_answers_are_unreadable(shape: str) -> None:
    routes = {LEG: Route(READ_BAD[shape])}
    spec, _ = spec_for(routes)
    aclient, _ = make_async_client(routes)
    aspec = LexAPIToolSpec(async_client=aclient)
    texts = await _all_paths(spec, aspec, "lexapi_search_legislation", {"query": "q"})
    assert all(WITH_STATUS in t for t in texts)


def _document(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = load_fixture("document.read.json")
    body["document"].update(overrides)
    return body


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (_document(), "[1] "),
        (_document(content=None), "LexAPI returned no text for CELEX 32099R0001."),
        (
            {"success": True, "document": {"celex": "32099R0001", "title": "Metadata only"}},
            "LexAPI returned no text for CELEX 32099R0001.",
        ),
        (_document(content={"fullText": "Whole text of a decision."}), "[1] Whole text"),
        (_document(content={"articles": []}), "LexAPI returned no text for CELEX 32099R0001."),
        (
            _document(
                content={
                    "fullText": "Article 1 ...",
                    "articles": [{"id": "art_17", "number": "Article 17", "content": "Text."}],
                    "recitals": [{"number": 1, "text": "Recital."}],
                    "sections": [],
                }
            ),
            "[1] Text.",
        ),
    ],
)
def test_well_formed_read_answers_are_accepted(body: dict[str, Any], expected: str) -> None:
    spec, _ = spec_for({DOC: Route(body)})
    assert spec.lexapi_read_document("32099R0001").startswith(expected)


@pytest.mark.parametrize(
    "fixture", ["as-of.resolved.json", "as-of.unresolved.json", "as-of.version-only.json"]
)
def test_every_as_of_outcome_is_accepted(fixture: str) -> None:
    spec, _ = spec_for({ASOF: fixture_route(fixture)})
    out = spec.lexapi_legislation_as_of("32099R0001", AS_OF, provision="art_17")
    assert "\nSource: CELEX 32099R0001" in out


def test_read_budget_counts_what_is_shown() -> None:
    body = _document(
        content={
            "articles": [
                {"id": "art_1", "content": " " * 50 + "a" * 150},
                {"id": "art_2", "content": "b" * 50},
            ]
        }
    )
    spec, _ = spec_for({DOC: Route(body)}, max_chars=200)
    with collect() as events:
        out = spec.lexapi_read_document("32099R0001")
    assert _shown_text_total(out) <= 200
    assert "b" * 50 not in out
    assert events[0].report.not_shown == 1


# ── arguments checked before any request ───────────────────────────────


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("lexapi_search_legislation", {"query": ""}, "Invalid argument: query must not be empty"),
        ("lexapi_search_legislation", {"query": "  "}, "Invalid argument: query must not be empty"),
        ("lexapi_read_document", {"celex": ""}, "Invalid argument: celex must not be empty"),
        (
            "lexapi_read_document",
            {"celex": "32099R0001", "language": "EN"},
            "Invalid argument: language must be a two-letter code such as en; got 'EN'",
        ),
        (
            "lexapi_legislation_as_of",
            {"celex": "", "as_of": AS_OF},
            "Invalid argument: celex must not be empty",
        ),
        (
            "lexapi_legislation_as_of",
            {"celex": "32099R0001", "as_of": AS_OF, "language": "deu"},
            "Invalid argument: language must be a two-letter code such as en; got 'deu'",
        ),
    ],
)
async def test_arguments_are_checked_before_any_request(
    name: str, args: dict[str, Any], message: str
) -> None:
    spec, requests = spec_for(ROUTES)
    aclient, arecorder = make_async_client(ROUTES)
    aspec = LexAPIToolSpec(async_client=aclient)
    with pytest.raises(LexAPIToolError) as info:
        getattr(spec, name)(**args)
    assert str(info.value) == message
    with pytest.raises(LexAPIToolError) as ainfo:
        await getattr(aspec, "a" + name)(**args)
    assert str(ainfo.value) == message
    assert requests == [] and arecorder.requests == []


def test_argument_descriptions_are_pinned() -> None:
    spec, _ = spec_for({})
    props = {
        n: t.metadata.get_parameters_dict()["properties"] for n, t in tools_by_name(spec).items()
    }
    assert props["lexapi_search_legislation"]["query"]["description"] == (
        "What to look for, in plain words."
    )
    assert props["lexapi_search_legislation"]["as_of"]["description"] == (
        "Optional date, YYYY-MM-DD: return the text for that date. Leave it out for the "
        "latest text."
    )
    assert props["lexapi_read_document"]["article"]["description"] == "art_17, 17 or 'Article 17'"
    assert props["lexapi_legislation_as_of"]["provision"]["description"] == (
        "Optional provision: a whole article (art_17, 17 or 'Article 17'), an annex (anx_III) "
        "or recitals (rct_26 or rct_26-28). Without it only the version is named."
    )


@pytest.mark.parametrize("article", ["art_17", "17", "Article 17"])
def test_article_forms_are_sent_as_given(article: str) -> None:
    spec, requests = spec_for({DOC: fixture_route("document.read.json")})
    spec.lexapi_read_document("32099R0001", article=article)
    assert json.loads(requests[0].content)["articleId"] == article


@pytest.mark.parametrize(
    "provision", ["art_17", "17", "Article 17", "anx_III", "rct_26", "rct_26-28"]
)
def test_provision_forms_are_sent_as_given(provision: str) -> None:
    spec, requests = spec_for({ASOF: fixture_route("as-of.resolved.json")})
    spec.lexapi_legislation_as_of("32099R0001", AS_OF, provision=provision)
    assert json.loads(requests[0].content)["provision"] == provision
