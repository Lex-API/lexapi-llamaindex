"""The README's examples run against the offline stub."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest
from lexapi_offline import fixture_route
from readme_runner import run_readme

import llama_index.tools.lexapi._client as client_module

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def scripted_llm() -> Any:
    """A function-calling mock LLM that ends the agent run at once."""
    mock = pytest.importorskip("llama_index.core.llms.mock")
    if not hasattr(mock, "MockFunctionCallingLLM"):
        pytest.skip("this llama-index-core has no MockFunctionCallingLLM")
    return mock.MockFunctionCallingLLM()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_readme_examples_run(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = {
        "/legislation/semantic": fixture_route("legislation.mixed.json"),
        "/legislation/as-of": fixture_route("as-of.resolved.json"),
    }
    namespace, printed = run_readme(
        README, routes, client_module, monkeypatch, extra={"my_llm": scripted_llm()}
    )
    assert "Source: CELEX 32099R0001, art_17" in printed
    assert namespace["spec"]._clients.closed
