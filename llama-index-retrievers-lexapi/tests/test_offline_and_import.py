"""The guard itself, import-time behaviour, and use inside a query engine
with a mock LLM."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import lexapi
import pytest
from lexapi_offline import (
    assert_offline_base_url,
    assert_offline_client,
    fixture_route,
    make_client,
)
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.retrievers.lexapi import LexAPILegislationRetriever


def test_helper_refuses_non_loopback_base_urls() -> None:
    for url in (
        "https://lex-api.com/api/v1",
        "http://127.0.0.1.example.com/api/v1",
        "http://localhost:9/api/v1",
    ):
        with pytest.raises(RuntimeError, match="refuse the base URL"):
            assert_offline_base_url(url)
        with pytest.raises(RuntimeError, match="refuse the base URL"):
            make_client({}, base_url=url)


def test_helper_refuses_a_real_client() -> None:
    with pytest.raises(RuntimeError, match="refuse the base URL"):
        assert_offline_client(lexapi.LexAPI("lex_x"))


@pytest.mark.filterwarnings("ignore:A test tried to use socket")
def test_real_sockets_are_blocked() -> None:
    import socket

    with pytest.raises(Exception, match="socket"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)


def test_settings_point_at_mock_models() -> None:
    assert isinstance(Settings.llm, MockLLM)
    assert isinstance(Settings.embed_model, MockEmbedding)


IMPORT_PROBE = textwrap.dedent(
    """
    import socket, sys
    def refuse(*a, **k):
        raise SystemExit("network at import")
    socket.socket.connect = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse
    sys.modules["openai"] = None
    import llama_index.retrievers.lexapi as m
    assert "openai" not in [k for k, v in sys.modules.items() if v is not None]
    assert not any(k.startswith("llama_index.llms.openai") for k in sys.modules)
    print(sorted(m.__all__))
    """
)


def test_import_needs_no_openai_and_no_network() -> None:
    out = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert "LexAPILegislationRetriever" in out.stdout


def test_inside_a_query_engine_the_citation_reaches_the_llm() -> None:
    client, _ = make_client({"/legislation/semantic": fixture_route("legislation.mixed.json")})
    retriever = LexAPILegislationRetriever(client, as_of="2099-06-01")
    engine = RetrieverQueryEngine.from_args(retriever, llm=MockLLM())
    response = engine.query("What does article 17 say?")
    prompt = str(response)
    assert "Source: CELEX 32099R0001, art_17" in prompt
    assert "no legal force" in prompt
    assert "UNRESOLVEDTEXTTOKEN" not in prompt
    assert "[No text: CELEX 32099L0003 art_12" in prompt
    assert len(response.source_nodes) == 3
