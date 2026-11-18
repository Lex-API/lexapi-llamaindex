"""The README's examples run against the offline stub."""

from __future__ import annotations

import pathlib

import pytest
from lexapi_offline import fixture_route
from readme_runner import run_readme

import llama_index.retrievers.lexapi._client as client_module

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def test_readme_examples_run(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = {"/legislation/semantic": fixture_route("legislation.mixed.json")}
    namespace, printed = run_readme(README, routes, client_module, monkeypatch)
    assert "32099R0001 CELEX 32099R0001, art_17" in printed
    assert "None CELEX" not in printed
    assert namespace["report"].returned == 3
    assert len(namespace["nodes"]) == 3
