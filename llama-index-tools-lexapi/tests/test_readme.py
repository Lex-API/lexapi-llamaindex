"""The README's examples run against the offline stub."""

from __future__ import annotations

import pathlib

import pytest
from lexapi_offline import fixture_route, scripted_llm
from readme_runner import core_at_least, run_readme

import llama_index.tools.lexapi._client as client_module

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
RESET = "2099-07-01T00:00:00.000Z"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_readme_examples_run(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = {
        "/legislation/semantic": fixture_route("legislation.mixed.json"),
        "/legislation/as-of": fixture_route("as-of.resolved.json"),
        # The agent examples read a document and get CREDITS_EXHAUSTED.
        "/documentContent": fixture_route("error.402.json", 402),
    }
    llm = scripted_llm("lexapi_read_document", {"celex": "32016R0679"})
    namespace, printed = run_readme(
        README, routes, client_module, monkeypatch, extra={"my_llm": llm}
    )
    assert "Source: CELEX 32099R0001, art_17" in printed
    assert f"CREDITS_EXHAUSTED {RESET} 0\n" in printed
    assert "call(): CREDITS_EXHAUSTED\n" in printed
    assert f"wrapped: CREDITS_EXHAUSTED {RESET}\n" in printed
    on_014 = "event: CREDITS_EXHAUSTED\nresult: CREDITS_EXHAUSTED\n" in printed
    assert on_014 is core_at_least(0, 14)
    assert namespace["spec"]._clients.closed
