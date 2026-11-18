"""Property test: under ``withhold`` and ``drop`` no token of an unresolved
hit's text or snippet reaches a node, in any view, over generated bodies."""

from __future__ import annotations

import json
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from lexapi import passages
from lexapi.models import SemanticLegislationResponse
from lexapi_offline import Route, legislation_body, legislation_hit, make_client
from llama_index.core.schema import MetadataMode, TextNode

from llama_index.retrievers.lexapi import LexAPILegislationRetriever

AS_OF = "2099-06-01"

hit_kind = st.sampled_from(
    ["resolved", "unresolved", "other_date", "no_flag", "undated", "no_text"]
)
hits_strategy = st.lists(
    st.tuples(hit_kind, st.booleans(), st.sampled_from(["fetch_timeout", "not_yet_in_force"])),
    min_size=0,
    max_size=8,
)


def build(spec: list[tuple[str, bool, str]]) -> tuple[list[dict[str, Any]], set[str]]:
    """Hits, and the tokens that must not appear (texts and snippets of hits
    not established for the date asked)."""
    hits: list[dict[str, Any]] = []
    forbidden: set[str] = set()
    for n, (kind, consolidated, reason) in enumerate(spec):
        if kind == "resolved":
            hit = legislation_hit(n, resolved=True, consolidated=consolidated)
        elif kind == "unresolved":
            hit = legislation_hit(n, resolved=False, consolidated=consolidated, reason=reason)
        elif kind == "other_date":
            hit = legislation_hit(
                n, resolved=True, consolidated=consolidated, echoed_as_of="2098-01-01"
            )
        elif kind == "no_text":
            hit = legislation_hit(n, resolved=False, with_text=False)
        elif kind == "no_flag":
            hit = legislation_hit(n, resolved=None, consolidated=consolidated)
        else:
            hit = legislation_hit(n, resolved=None, consolidated=consolidated)
        if kind != "resolved":
            forbidden.update({f"TXT{n}Q", f"SNP{n}Q"})
        hits.append(hit)
    return hits, forbidden


@settings(max_examples=600, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(spec=hits_strategy, policy=st.sampled_from(["withhold", "drop"]), echo=st.booleans())
def test_no_unresolved_token_reaches_any_node(
    spec: list[tuple[str, bool, str]], policy: Any, echo: bool
) -> None:
    hits, forbidden = build(spec)
    body = legislation_body(hits, AS_OF if echo else None)
    client, _ = make_client({"/legislation/semantic": Route(body)})
    retriever = LexAPILegislationRetriever(
        client, as_of=AS_OF, unresolved=policy, similarity_top_k=8
    )
    got = retriever.retrieve_with_result("q")
    expected = passages.from_legislation_search(
        SemanticLegislationResponse(body), unresolved=policy, as_of=AS_OF
    )
    assert got.result == expected
    for item in got.nodes:
        node = item.node
        assert isinstance(node, TextNode)
        views = [
            node.text,
            json.dumps(node.metadata),
            node.get_content(metadata_mode=MetadataMode.ALL),
            node.get_content(metadata_mode=MetadataMode.LLM),
            node.get_content(metadata_mode=MetadataMode.EMBED),
            repr(node),
        ]
        for view in views:
            for token in forbidden:
                assert token not in view
        assert node.get_content(metadata_mode=MetadataMode.EMBED) == node.text
    if policy == "drop":
        assert all(not item.node.metadata["text_withheld"] for item in got.nodes)
