"""Offline test setup: no network, no LLM, no embeddings.

The shared harness lives in ``testing/`` at the repository root.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Iterator

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "testing"))

import lexapi_offline


@pytest.fixture(autouse=True, scope="session")
def _offline_session() -> Iterator[None]:
    with lexapi_offline.network_guard():
        lexapi_offline.disable_llm_and_embeddings()
        yield
