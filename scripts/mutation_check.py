"""Apply deliberate faults to the packages and check the tests catch each one.

Usage (from the repository root, in an environment where both packages are
installed in editable mode with their dev dependencies)::

    python scripts/mutation_check.py

Each mutant edits one source file in place, runs that package's tests and
restores the file. A mutant that leaves the tests green is reported as
SURVIVED and the script exits non-zero.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
RET = "llama-index-retrievers-lexapi"
TOOLS = "llama-index-tools-lexapi"
RET_BASE = f"{RET}/llama_index/retrievers/lexapi/base.py"
RET_NODES = f"{RET}/llama_index/retrievers/lexapi/_nodes.py"
TOOLS_BASE = f"{TOOLS}/llama_index/tools/lexapi/base.py"


@dataclass(frozen=True)
class Mutant:
    name: str
    path: str
    old: str
    new: str

    @property
    def package(self) -> str:
        return self.path.split("/", 1)[0]


MUTANTS = [
    Mutant(
        "node text bypasses the core (raw text)",
        RET_NODES,
        "return passages.format_passage(shown)",
        'return shown.text or ""',
    ),
    Mutant(
        "as_of not passed to from_legislation_search",
        RET_BASE,
        "response, unresolved=self._unresolved, as_of=self._as_of",
        "response, unresolved=self._unresolved",
    ),
    Mutant(
        "error turned into []",
        RET_BASE,
        "        result = self._search(query_bundle.query_str)\n        return self._finish(",
        "        try:\n            result = self._search(query_bundle.query_str)\n"
        "        except Exception:\n            return []\n        return self._finish(",
    ),
    Mutant(
        "content() by default (no citation in text)",
        RET_BASE,
        '        unresolved: UnresolvedPolicy = "withhold",\n        emit_withheld: bool = True,\n'
        "        include_citation_in_text: bool = True,",
        '        unresolved: UnresolvedPolicy = "withhold",\n        emit_withheld: bool = True,\n'
        "        include_citation_in_text: bool = False,",
    ),
    Mutant(
        "hand-built metadata",
        RET_NODES,
        "    metadata = passage.metadata()",
        '    metadata = {"celex": passage.celex, "citation": passages.format_citation(passage)}',
    ),
    Mutant(
        "over-asking when celex is set",
        RET_BASE,
        '            "limit": self._similarity_top_k,\n            "include_text": True,',
        '            "limit": min(50, self._similarity_top_k * 4)'
        " if self._celex else self._similarity_top_k,\n"
        '            "include_text": True,',
    ),
    Mutant(
        "case-law snippet used as text",
        RET_BASE,
        "        return passages.from_case_law_search(response)",
        "        import dataclasses\n"
        "        r = passages.from_case_law_search(response)\n"
        "        return dataclasses.replace(r, passages=tuple(dataclasses.replace("
        'p, text=p.text or p.snippet, text_status="provided") for p in r.passages))',
    ),
    Mutant(
        "guardrail key hidden from the LLM",
        RET_NODES,
        '("eli", "id", "kind", "score", "url")',
        '("as_of_reason", "eli", "id", "kind", "score", "url")',
    ),
    Mutant(
        "API key kept on the retriever",
        RET_BASE,
        "        self._language = language\n        self._min_score = min_score\n"
        "        self._celex = _celex_tuple(celex)",
        "        self._language = language\n        self._api_key = api_key\n"
        "        self._min_score = min_score\n        self._celex = _celex_tuple(celex)",
    ),
    Mutant(
        "metadata embedded",
        RET_NODES,
        "excluded_embed_metadata_keys=sorted(metadata),",
        "excluded_embed_metadata_keys=[],",
    ),
    Mutant(
        "score changed",
        RET_NODES,
        "return NodeWithScore(node=node, score=passage.score)",
        "return NodeWithScore(node=node, score=1.0)",
    ),
    Mutant(
        "legislation search without include_text",
        RET_BASE,
        '            "limit": self._similarity_top_k,\n            "include_text": True,',
        '            "limit": self._similarity_top_k,\n            "include_text": False,',
    ),
    Mutant(
        "emit_withheld ignored",
        RET_NODES,
        "if passage.text_withheld and not emit_withheld:",
        "if False:",
    ),
    Mutant(
        "default policy not withhold",
        RET_BASE,
        '        unresolved: UnresolvedPolicy = "withhold",\n        emit_withheld',
        '        unresolved: UnresolvedPolicy = "keep_labelled",\n        emit_withheld',
    ),
    Mutant(
        "last result stored on the instance",
        RET_BASE,
        "            holder.setdefault(id(self), result)",
        "            holder.setdefault(id(self), result)\n        self._last_result = result",
    ),
    Mutant(
        "duplicate id: later passage wins",
        RET_NODES,
        "if current is None or (passage.has_text and not current.has_text):",
        "if True:",
    ),
    Mutant(
        "truncation not marked",
        RET_NODES,
        'return dataclasses.replace(passage, text=f"{kept}\\n{marker}")',
        "return dataclasses.replace(passage, text=kept)",
    ),
    Mutant(
        "case law sends hyde",
        RET_BASE,
        '            "filters": dict(self._filters) if self._filters else None,\n        }',
        '            "filters": dict(self._filters) if self._filters else None,\n'
        '            "hyde": True,\n        }',
    ),
    Mutant(
        "async path uses the sync client",
        RET_BASE,
        "        client = self._clients.async_\n        if client is None:",
        "        client = self._clients.async_\n        if True:",
    ),
    Mutant(
        "celex restriction skipped",
        RET_BASE,
        "        if self._celex is not None:\n            result = passages.restrict_to_celex",
        "        if False:\n            result = passages.restrict_to_celex",
    ),
    # ── tools ──
    Mutant(
        "tool error swallowed",
        TOOLS_BASE,
        "    try:\n        return fn()\n    except LexAPIError as error:\n"
        "        raise LexAPIToolError.from_sdk(error) from error",
        "    try:\n        return fn()\n    except LexAPIError:\n"
        "        return None  # type: ignore",
    ),
    Mutant(
        "key masking removed",
        TOOLS_BASE,
        '_KEY_PATTERN.sub("lex_***", message),',
        "message,",
    ),
    Mutant(
        "tool search: as_of not passed to the core",
        TOOLS_BASE,
        "response, unresolved=self._unresolved, as_of=as_of",
        "response, unresolved=self._unresolved",
    ),
    Mutant(
        "tool output bypasses format_passage",
        TOOLS_BASE,
        "blocks = [passages.format_passage(p) for p in result.passages]",
        "blocks = [p.content() for p in result.passages]",
    ),
    Mutant(
        "error code missing from tool error",
        TOOLS_BASE,
        'message = f"LexAPI error {error.code}{status}: {error.message}"',
        'message = f"LexAPI error{status}: {error.message}"',
    ),
    Mutant(
        "law on a date ignores the unresolved policy",
        TOOLS_BASE,
        "response, provision=provision, unresolved=self._unresolved",
        "response, provision=provision",
    ),
    Mutant(
        "tool search over-asks",
        TOOLS_BASE,
        '            "limit": self._similarity_top_k,',
        '            "limit": self._similarity_top_k * 2,',
    ),
    Mutant(
        "tool description claims more than the core",
        TOOLS_BASE,
        '"""Search EU legislation by meaning; returns matching articles, each with its citation.',
        '"""Search EU legislation in force by meaning; returns matching articles, each with its'
        " citation.",
    ),
]


def run(mutant: Mutant, python: str) -> bool:
    """True when the tests caught the mutant."""
    path = ROOT / mutant.path
    original = path.read_text(encoding="utf-8")
    if original.count(mutant.old) != 1:
        raise SystemExit(f"mutant {mutant.name!r}: anchor found {original.count(mutant.old)} times")
    path.write_text(original.replace(mutant.old, mutant.new), encoding="utf-8")
    try:
        result = subprocess.run(
            [python, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
            cwd=ROOT / mutant.package,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        path.write_text(original, encoding="utf-8")
    return result.returncode != 0


def main() -> int:
    python = sys.executable
    survived = 0
    for mutant in MUTANTS:
        caught = run(mutant, python)
        survived += not caught
        print(f"{'caught  ' if caught else 'SURVIVED'}  {mutant.package:32} {mutant.name}")
    print(f"{len(MUTANTS) - survived}/{len(MUTANTS)} mutants caught")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
