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
#: (``node.get_content(MetadataMode.LLM)``). The rest stays visible, in
#: particular ``citation``, ``cite_celex``, ``consolidated``, ``as_of``,
#: ``as_of_resolved``, ``as_of_reason`` and ``text_status``.
#:
#: - ``score`` and ``id``: retrieval bookkeeping, not part of the source.
#: - ``url``: already the last part of ``citation``.
#: - ``kind``: which LexAPI call produced the passage.
#: - ``eli``: a second identifier of the document; ``citation`` names the
#:   CELEX number to cite.
EXCLUDED_LLM_METADATA_KEYS: tuple[str, ...] = ("eli", "id", "kind", "score", "url")

#: Smallest accepted ``max_chars``.
MIN_MAX_CHARS = 200


@dataclass(frozen=True)
class LexAPICallReport:
    """What one call did.

    ``requested`` is the number of hits asked for and ``returned`` the number
    of nodes returned. ``withheld``, ``kept_labelled``, ``dropped``,
    ``uncitable`` and ``filtered_out`` are the counters of
    :class:`lexapi.passages.PassageResult`, counted before de-duplication.
    ``withheld_not_emitted`` counts withheld passages left out by
    ``emit_withheld=False``; ``returned_withheld`` the withheld nodes
    returned; ``duplicates`` the passages merged into another of the same
    id; ``truncated`` the texts cut by ``max_chars``. ``units_charged`` and
    ``credits_remaining`` are as the response reported them (``None`` when
    it reported none).
    """

    requested: int
    returned: int
    as_of: str | None
    withheld: int
    kept_labelled: int
    dropped: int
    uncitable: int
    filtered_out: int
    withheld_not_emitted: int
    returned_withheld: int
    duplicates: int
    truncated: int
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


def truncate(passage: Passage, max_chars: int) -> tuple[Passage, bool]:
    """The passage with its text cut to at most ``max_chars`` characters,
    and whether it was cut.

    Leading whitespace is dropped first. The cut is made at the last blank
    line within the limit, else the last line break, else the last space,
    else at the limit, and never leaves an empty text. The cut is marked in
    the text.
    """
    text = passage.text
    if text is None:
        return passage, False
    text = text.lstrip()
    if len(text) <= max_chars:
        return passage, False
    cut = 0
    for separator in ("\n\n", "\n", " "):
        cut = text.rfind(separator, 0, max_chars + 1)
        if cut > 0 and text[:cut].strip():
            break
        cut = 0
    if cut <= 0:
        cut = max_chars
    kept = text[:cut].rstrip()
    note = (
        f"[Text cut by {DISTRIBUTION}: {len(kept)} of {len(text)} characters shown. "
        "The full text is at the source.]"
    )
    return dataclasses.replace(passage, text=f"{kept}\n{note}"), True


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
        excluded_llm_metadata_keys=list(EXCLUDED_LLM_METADATA_KEYS),
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
        withheld_not_emitted=skipped,
        returned_withheld=sum(p.text_withheld for p in kept),
        duplicates=duplicates,
        truncated=truncated,
        units_charged=result.units_charged,
        credits_remaining=result.credits_remaining,
        hint=result.hint,
    )
    return nodes, report
