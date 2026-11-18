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
        "        return passages.format_passage(passage)",
        '        return passage.text or ""',
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
        "    metadata = (metadata_from or passage).metadata()",
        '    metadata = {"celex": passage.celex, "citation": passages.format_citation(passage)}',
    ),
    Mutant(
        "over-asking when celex is set",
        RET_BASE,
        '            "limit": self._similarity_top_k,\n            "include_text": True,\n'
        '            "language": self._language,\n            "min_score": self._min_score,\n'
        '            "as_of"',
        '            "limit": min(50, self._similarity_top_k * 4)'
        " if self._celex else self._similarity_top_k,\n"
        '            "include_text": True,\n'
        '            "language": self._language,\n            "min_score": self._min_score,\n'
        '            "as_of"',
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
        '            "include_text": True,\n            "language": self._language,\n'
        '            "min_score": self._min_score,\n            "as_of"',
        '            "include_text": False,\n            "language": self._language,\n'
        '            "min_score": self._min_score,\n            "as_of"',
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
        "            holder.setdefault(id(self), (report, result))",
        "            holder.setdefault(id(self), (report, result))\n"
        "        self._last_report = report",
    ),
    Mutant(
        "duplicate id: later passage wins",
        RET_NODES,
        "        if _rank(passage) > _rank(current):",
        "        if True:",
    ),
    Mutant(
        "truncation not marked",
        RET_NODES,
        'return dataclasses.replace(passage, text=f"{kept}\\n{note}"), True',
        "return dataclasses.replace(passage, text=kept), True",
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
        '    return _KEY_PATTERN.sub("lex_***", text)',
        "    return text",
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
        "        else:\n            block = passages.format_passage(candidate)",
        "        else:\n            block = candidate.content()",
    ),
    Mutant(
        "error code missing from tool error",
        TOOLS_BASE,
        'message = f"LexAPI error {error.code}{status}: {api_message}"',
        'message = f"LexAPI error{status}: {api_message}"',
    ),
    Mutant(
        "law on a date ignores the unresolved policy",
        TOOLS_BASE,
        '            response, provision=provision, unresolved="withhold"',
        "            response, provision=provision, unresolved=self._unresolved",
    ),
    Mutant(
        "tool search over-asks",
        TOOLS_BASE,
        '            "limit": self._similarity_top_k,',
        '            "limit": self._similarity_top_k * 2,',
    ),
    Mutant(
        "case law without include_text",
        RET_BASE,
        '            "include_text": True,\n            "language": self._language,\n'
        '            "min_score": self._min_score,\n            "filters"',
        '            "include_text": False,\n            "language": self._language,\n'
        '            "min_score": self._min_score,\n            "filters"',
    ),
    Mutant(
        "report counts passages before de-dup as returned",
        RET_NODES,
        "        returned=len(nodes),",
        "        returned=len(result.passages),",
    ),
    Mutant(
        "async client built eagerly",
        f"{RET}/llama_index/retrievers/lexapi/_client.py",
        "        self._async = None\n        self._owned = True",
        "        self._async = AsyncLexAPI(api_key, **self._kwargs)\n        self._owned = True",
    ),
    Mutant(
        "billing URL missing from the past-due error",
        TOOLS_BASE,
        '            message += f" Billing: {error.billing_url}"',
        "            pass",
    ),
    Mutant(
        "period rule ignored",
        TOOLS_BASE,
        '        if api_message and api_message[-1] not in ".!?":',
        "        if api_message:",
    ),
    Mutant(
        "description ignores the unresolved policy",
        TOOLS_BASE,
        "tool_metadata(self._unresolved)",
        "tool_metadata()",
    ),
    Mutant(
        "read budget ignored",
        TOOLS_BASE,
        "            max_chars=self._max_chars,",
        "            max_chars=None,",
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
    for mutant in MUTANTS:
        count = (ROOT / mutant.path).read_text(encoding="utf-8").count(mutant.old)
        if count != 1:
            raise SystemExit(f"mutant {mutant.name!r}: anchor found {count} times")
    survived = 0
    for mutant in MUTANTS:
        caught = run(mutant, python)
        survived += not caught
        print(f"{'caught  ' if caught else 'SURVIVED'}  {mutant.package:32} {mutant.name}")
    print(f"{len(MUTANTS) - survived}/{len(MUTANTS)} mutants caught")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
