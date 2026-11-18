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
TOOLS_PASSAGES = f"{TOOLS}/llama_index/tools/lexapi/_passages.py"


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
        'EXCLUDED_LLM_METADATA_KEYS: tuple[str, ...] = (\n    "base_celex",',
        'EXCLUDED_LLM_METADATA_KEYS: tuple[str, ...] = (\n    "as_of_reason",\n    "base_celex",',
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
        '    return f"{kept}\\n{note}", True',
        "    return kept, True",
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
    # ── retrievers: cut rule, dates, report, lifecycle ──
    Mutant(
        "cut rule: no space fallback (retrievers)",
        RET_NODES,
        '    for separator in ("\\n\\n", "\\n", " "):',
        '    for separator in ("\\n\\n", "\\n"):',
    ),
    Mutant(
        "cut note counts the stripped text",
        RET_NODES,
        "{len(kept)} of {len(text)} characters shown",
        "{len(kept)} of {len(body)} characters shown",
    ),
    Mutant(
        "retriever date not validated",
        RET_BASE,
        '    raise ValueError(f"{name} must be a date in the form YYYY-MM-DD; got {value!r}")',
        "    return value  # type: ignore[return-value]",
    ),
    Mutant(
        "report field order changed",
        RET_NODES,
        "    returned_withheld: int\n    returned_labelled: int\n",
        "    returned_labelled: int\n    returned_withheld: int\n",
    ),
    Mutant(
        "closed instance still answers",
        f"{RET}/llama_index/retrievers/lexapi/_client.py",
        '            raise ClientClosedError(f"{self._owner} is closed; create a new one.")',
        "            pass",
    ),
    Mutant(
        "only-async misuse is a TypeError",
        f"{RET}/llama_index/retrievers/lexapi/_client.py",
        '            raise ValueError(\n                f"{self._owner} was given only',
        '            raise TypeError(\n                f"{self._owner} was given only',
    ),
    # ── tools ──
    Mutant(
        "tool error swallowed",
        TOOLS_BASE,
        "        return fn()\n    except LexAPIError as error:\n"
        "        failure = LexAPIToolError.from_sdk(error)",
        "        return fn()\n    except LexAPIError:\n        return None  # type: ignore",
    ),
    Mutant(
        "SDK error chained to the tool error",
        TOOLS_BASE,
        "        return fn()\n    except LexAPIError as error:\n"
        "        failure = LexAPIToolError.from_sdk(error)",
        "        return fn()\n    except LexAPIError as error:\n"
        "        raise LexAPIToolError.from_sdk(error) from None",
    ),
    Mutant(
        "unreadable answer passes the SDK message on",
        TOOLS_BASE,
        "        if isinstance(error, InvalidResponseError) or (",
        "        if False and (",
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
        'f"[{n}] {passages.format_passage(p)}" if numbered',
        'f"[{n}] {p.content()}" if numbered',
    ),
    Mutant(
        "output not numbered",
        TOOLS_BASE,
        'f"[{n}] {passages.format_passage(p)}" if numbered',
        'f"{passages.format_passage(p)}" if numbered',
    ),
    Mutant(
        "drop note only when nothing is shown",
        TOOLS_BASE,
        "    if result.dropped:\n",
        "    if result.dropped and not shown:\n",
    ),
    Mutant(
        "error code missing from tool error",
        TOOLS_BASE,
        'message = f"LexAPI error {error.code}{http}: {api_message}"',
        'message = f"LexAPI error{http}: {api_message}"',
    ),
    Mutant(
        "retry wait only for 429",
        TOOLS_BASE,
        "        if error.retry_after is not None:",
        "        if error.status == 429 and error.retry_after is not None:",
    ),
    Mutant(
        "as-of tool uses the search policy",
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
        "description ignores the unresolved policy",
        TOOLS_BASE,
        "**tool_metadata(self._unresolved, self._max_chars),",
        '**tool_metadata("withhold", self._max_chars),',
    ),
    Mutant(
        "read budget ignored",
        TOOLS_BASE,
        "max_chars=self._max_chars, celex=celex",
        "max_chars=None, celex=celex",
    ),
    Mutant(
        "read budget counts Source lines",
        TOOLS_BASE,
        "        length = _body_length(passage)",
        "        length = len(passages.format_passage(passage))",
    ),
    Mutant(
        "cut rule: no space fallback (tools)",
        "llama-index-tools-lexapi/llama_index/tools/lexapi/_passages.py",
        '    for separator in ("\\n\\n", "\\n", " "):',
        '    for separator in ("\\n\\n", "\\n"):',
    ),
    Mutant(
        "tool date not validated",
        TOOLS_BASE,
        '    raise LexAPIToolError.invalid_argument(\n        f"{name} must be a date',
        '    return\n    raise LexAPIToolError.invalid_argument(\n        f"{name} must be a date',
    ),
    Mutant(
        "as-of schema without the date pattern",
        TOOLS_BASE,
        '    as_of: str = Field(pattern=_DATE_PATTERN, description="The date, YYYY-MM-DD.")',
        '    as_of: str = Field(description="The date, YYYY-MM-DD.")',
    ),
    Mutant(
        "not_shown not reported",
        TOOLS_BASE,
        "        not_shown=len(rest),",
        "        not_shown=0,",
    ),
    Mutant(
        "read reports a requested count",
        TOOLS_BASE,
        "            result, requested=None, empty=empty, max_chars=self._max_chars, celex=celex",
        "            result, requested=1, empty=empty, max_chars=self._max_chars, celex=celex",
    ),
    Mutant(
        "read-without-text string changed",
        TOOLS_BASE,
        'f"LexAPI returned no text for article {article} of CELEX {celex}."',
        'f"LexAPI returned no text for CELEX {celex} {article}."',
    ),
    Mutant(
        "read without article omits fullText",
        TOOLS_BASE,
        '["metadata", "articles", "fullText"]',
        "None",
    ),
    Mutant(
        "tool description claims more than the core",
        TOOLS_BASE,
        '"Search EU legislation (EUR-Lex) by meaning. Returns matching articles, each followed "',
        '"Search EU legislation in force (EUR-Lex) by meaning. Returns matching articles, "',
    ),
    # ── malformed answers, read budget, cut rule, arguments, LLM view ──
    Mutant(
        "read answer shape not checked",
        TOOLS_BASE,
        "        _check_document_shape(response)\n",
        "",
    ),
    Mutant(
        "as-of answer shape not checked",
        TOOLS_BASE,
        "        _check_as_of_shape(response)\n",
        "",
    ),
    Mutant(
        "as-of resolved flag not checked",
        TOOLS_BASE,
        '    if not isinstance(response.get("asOfResolved"), bool):',
        "    if False:",
    ),
    Mutant(
        "articles not checked as objects",
        TOOLS_BASE,
        "not all(isinstance(a, dict) for a in articles)",
        "False",
    ),
    Mutant(
        "read budget counts the stripped text",
        TOOLS_BASE,
        "    return len(passage.content())",
        "    return len((passage.text or passage.content()).lstrip())",
    ),
    Mutant(
        "cut rule takes a break in the first half (retrievers)",
        RET_NODES,
        "        if at > 0 and max_chars // 2 <= at <= max_chars:",
        "        if at > 0:",
    ),
    Mutant(
        "cut rule takes a break in the first half (tools)",
        TOOLS_PASSAGES,
        "        if at > 0 and max_chars // 2 <= at <= max_chars:",
        "        if at > 0:",
    ),
    Mutant(
        "cut rule strips a text that fits",
        RET_NODES,
        "    if len(text) <= max_chars:\n        return text, False",
        "    if len(text.lstrip()) <= max_chars:\n        return text.lstrip(), False",
    ),
    Mutant(
        "empty query sent",
        TOOLS_BASE,
        '        _check_not_empty(query, "query")',
        "        pass",
    ),
    Mutant(
        "read language not checked",
        TOOLS_BASE,
        '        _check_not_empty(celex, "celex")\n        _check_language(language)\n\n',
        '        _check_not_empty(celex, "celex")\n\n',
    ),
    Mutant(
        "citation shown twice to the LLM",
        RET_NODES,
        '    "celex",\n    "citation",\n    "eli",',
        '    "celex",\n    "eli",',
    ),
    Mutant(
        "citation hidden without the Source line",
        RET_NODES,
        'if include_citation_in_text or key != "citation"',
        "if True",
    ),
    Mutant(
        "closed tool spec reaches the client",
        TOOLS_BASE,
        "        if self._clients.closed:",
        "        if False:",
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
