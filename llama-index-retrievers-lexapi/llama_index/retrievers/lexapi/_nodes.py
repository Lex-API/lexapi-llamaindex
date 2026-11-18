"""Passages to LlamaIndex nodes, and the per-call report.

Every node is built from a :class:`lexapi.passages.Passage`: the text is
:func:`lexapi.passages.format_passage` (or :meth:`Passage.content`), the id
is :attr:`Passage.id` and the metadata is :meth:`Passage.metadata`, unchanged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore, TextNode

from lexapi import passages
from lexapi.passages import Passage, PassageResult

DISTRIBUTION = "llama-index-retrievers-lexapi"

#: Metadata keys hidden from the LLM view of a node
#: (``node.get_content(MetadataMode.LLM)``), so the model reads each fact
#: once: ``cite_celex``, ``consolidated``, ``consolidated_note``, the
#: ``as_of`` fields and ``text_status`` stay visible, and the citation is
#: read once, in the node text's ``Source:`` line. With
#: ``include_citation_in_text=False`` the node text has no ``Source:`` line,
#: so ``citation`` is not hidden.
#:
#: - ``citation``: the ``Source:`` line of the node text.
#: - ``celex``, ``base_celex``, ``version_*``: in ``citation``; the number to
#:   cite is ``cite_celex``.
#: - ``text_withheld``: the same fact as ``text_status``.
#: - ``score``, ``id``: retrieval bookkeeping, not part of the source.
#: - ``url``: the last part of ``citation``.
#: - ``kind``: which LexAPI call produced the passage.
#: - ``eli``: a second identifier of the document.
EXCLUDED_LLM_METADATA_KEYS: tuple[str, ...] = (
    "base_celex",
    "celex",
    "citation",
    "eli",
    "id",
    "kind",
    "score",
    "text_withheld",
    "url",
    "version_celex",
    "version_date",
    "version_kind",
    "version_valid_until",
)

#: Smallest accepted ``max_chars``.
MIN_MAX_CHARS = 200


@dataclass(frozen=True)
class LexAPICallReport:
    """What one call did.

    - ``requested``: hits asked for (``None`` when the call asks for no
      number of hits).
    - ``returned``: nodes (or passages) returned.
    - ``as_of``: the date asked for.
    - ``withheld``, ``kept_labelled``, ``dropped``, ``uncitable``,
      ``filtered_out``: the counters of :class:`lexapi.passages.PassageResult`,
      counted before de-duplication.
    - ``returned_withheld`` / ``returned_labelled``: returned passages whose
      text was withheld / is a labelled latest text.
    - ``withheld_not_emitted``: withheld passages left out by
      ``emit_withheld=False``.
    - ``duplicates``: passages merged into another of the same id.
    - ``truncated``: returned texts that were cut.
    - ``not_shown``: passages left out by a character budget.
    - ``units_charged``, ``credits_remaining``: as the response reported them
      (``None`` when it reported none).
    - ``hint``: the API's low-confidence hint.
    """

    requested: int | None
    returned: int
    as_of: str | None
    withheld: int
    kept_labelled: int
    dropped: int
    uncitable: int
    filtered_out: int
    returned_withheld: int
    returned_labelled: int
    withheld_not_emitted: int
    duplicates: int
    truncated: int
    not_shown: int
    units_charged: int | None
    credits_remaining: int | None
    hint: str | None


def _rank(passage: Passage) -> int:
    if passage.text_status == "provided":
        return 2
    if passage.text_status == "not_established_for_date":
        return 1
    return 0


def dedupe(
    items: Iterable[Passage], *, emit_withheld: bool = True
) -> tuple[list[Passage], int, int]:
    """Passages in API order, one per :attr:`Passage.id`, and the numbers
    of withheld passages skipped and of duplicates merged.

    Withheld passages are skipped first when ``emit_withheld`` is false.
    Each id keeps its first position; a later passage of the same id
    replaces the earlier one only when it ranks strictly higher: a text
    (``provided``) over a labelled latest text (``not_established_for_date``)
    over a notice.
    """
    chosen: dict[str, Passage] = {}
    skipped = duplicates = 0
    for passage in items:
        if passage.text_withheld and not emit_withheld:
            skipped += 1
            continue
        current = chosen.get(passage.id)
        if current is None:
            chosen[passage.id] = passage
            continue
        duplicates += 1
        if _rank(passage) > _rank(current):
            chosen[passage.id] = passage
    return list(chosen.values()), skipped, duplicates


def cut_text(text: str, max_chars: int) -> tuple[str, bool]:
    """``text`` within ``max_chars`` characters, and whether it was cut.

    A text of at most ``max_chars`` characters is returned unchanged.
    Otherwise leading whitespace is dropped and the text is cut at a
    paragraph break when one falls in the second half of the limit,
    otherwise at a line break or a space there, otherwise at the limit;
    a note saying so follows.
    """
    if len(text) <= max_chars:
        return text, False
    body = text.lstrip()
    if len(body) <= max_chars:
        return body, False
    cut = max_chars
    for separator in ("\n\n", "\n", " "):
        at = body.rfind(separator, 0, max_chars + len(separator))
        if at > 0 and max_chars // 2 <= at <= max_chars:
            cut = at
            break
    kept = body[:cut].rstrip() or body[:max_chars]
    note = (
        f"[Text cut by {DISTRIBUTION}: {len(kept)} of {len(text)} characters shown. "
        "The full text is at the source.]"
    )
    return f"{kept}\n{note}", True


def truncate(passage: Passage, max_chars: int) -> tuple[Passage, bool]:
    """The passage with its text passed through :func:`cut_text`, and
    whether it was cut. A passage without text is returned unchanged."""
    if passage.text is None:
        return passage, False
    text, cut = cut_text(passage.text, max_chars)
    if text == passage.text:
        return passage, False
    return dataclasses.replace(passage, text=text), cut


def passage_text(passage: Passage, *, include_citation_in_text: bool = True) -> str:
    """The node text: :func:`format_passage` (text or notice, then the
    ``Source:`` line), or :meth:`Passage.content` alone."""
    if include_citation_in_text:
        return passages.format_passage(passage)
    return passage.content()


def passage_to_node(
    passage: Passage, *, include_citation_in_text: bool = True, metadata_from: Passage | None = None
) -> NodeWithScore:
    """One passage as a ``NodeWithScore``; the score is the passage's own."""
    metadata = (metadata_from or passage).metadata()
    node = TextNode(
        text=passage_text(passage, include_citation_in_text=include_citation_in_text),
        id_=passage.id,
        metadata=metadata,
        # Nothing of the metadata is embedded: it describes the citation,
        # not the text.
        excluded_embed_metadata_keys=sorted(metadata),
        excluded_llm_metadata_keys=[
            key
            for key in EXCLUDED_LLM_METADATA_KEYS
            if include_citation_in_text or key != "citation"
        ],
    )
    return NodeWithScore(node=node, score=passage.score)


def build(
    result: PassageResult,
    *,
    requested: int,
    include_citation_in_text: bool = True,
    emit_withheld: bool = True,
    max_chars: int | None = None,
) -> tuple[list[NodeWithScore], LexAPICallReport]:
    """The nodes of one result and the call's report."""
    kept, skipped, duplicates = dedupe(result.passages, emit_withheld=emit_withheld)
    nodes: list[NodeWithScore] = []
    truncated = 0
    for passage in kept:
        shown = passage
        if max_chars is not None:
            shown, cut = truncate(passage, max_chars)
            truncated += cut
        nodes.append(
            passage_to_node(
                shown, include_citation_in_text=include_citation_in_text, metadata_from=passage
            )
        )
    report = LexAPICallReport(
        requested=requested,
        returned=len(nodes),
        as_of=result.as_of,
        withheld=result.withheld,
        kept_labelled=result.kept_labelled,
        dropped=result.dropped,
        uncitable=result.uncitable,
        filtered_out=result.filtered_out,
        returned_withheld=sum(p.text_withheld for p in kept),
        returned_labelled=sum(p.text_status == "not_established_for_date" for p in kept),
        withheld_not_emitted=skipped,
        duplicates=duplicates,
        truncated=truncated,
        not_shown=0,
        units_charged=result.units_charged,
        credits_remaining=result.credits_remaining,
        hint=result.hint,
    )
    return nodes, report
