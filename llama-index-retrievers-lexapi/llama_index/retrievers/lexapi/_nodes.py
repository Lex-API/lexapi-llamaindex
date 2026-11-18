"""Passages to LlamaIndex nodes.

Every node is built from a :class:`lexapi.passages.Passage`: the text is
:func:`lexapi.passages.format_passage` (or :meth:`Passage.content`), the id
is :attr:`Passage.id` and the metadata is :meth:`Passage.metadata`, unchanged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from llama_index.core.schema import NodeWithScore, TextNode

from lexapi import passages
from lexapi.passages import Passage

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


def passage_text(
    passage: Passage, *, include_citation_in_text: bool = True, max_chars: int | None = None
) -> str:
    """The node text for a passage.

    ``include_citation_in_text=True`` gives :func:`format_passage`: the text
    or notice, then a ``Source:`` line with the citation. ``False`` gives
    :meth:`Passage.content` alone.
    """
    shown = passage if max_chars is None else truncate(passage, max_chars)
    if include_citation_in_text:
        return passages.format_passage(shown)
    return shown.content()


def passage_to_node(
    passage: Passage, *, include_citation_in_text: bool = True, max_chars: int | None = None
) -> NodeWithScore:
    """One passage as a ``NodeWithScore``; the score is the passage's own."""
    metadata = passage.metadata()
    node = TextNode(
        text=passage_text(
            passage, include_citation_in_text=include_citation_in_text, max_chars=max_chars
        ),
        id_=passage.id,
        metadata=metadata,
        # Nothing of the metadata is embedded: a node of this package is a
        # search result, and its metadata describes the citation, not the text.
        excluded_embed_metadata_keys=sorted(metadata),
        excluded_llm_metadata_keys=list(EXCLUDED_LLM_METADATA_KEYS),
    )
    return NodeWithScore(node=node, score=passage.score)


def passages_to_nodes(
    items: Iterable[Passage],
    *,
    include_citation_in_text: bool = True,
    emit_withheld: bool = True,
    max_chars: int | None = None,
) -> list[NodeWithScore]:
    """Nodes for ``items``, in order, one per :attr:`Passage.id`.

    :attr:`Passage.id` is not unique within a result (two hits of one
    decision without a paragraph number share it). The first passage of an
    id is kept, except that a passage with text replaces an earlier one of
    the same id without text, so a notice never stands in for a text.
    ``emit_withheld=False`` leaves out passages whose text was withheld.
    """
    chosen: dict[str, Passage] = {}
    for passage in items:
        if passage.text_withheld and not emit_withheld:
            continue
        current = chosen.get(passage.id)
        if current is None or (passage.has_text and not current.has_text):
            chosen[passage.id] = passage
    return [
        passage_to_node(p, include_citation_in_text=include_citation_in_text, max_chars=max_chars)
        for p in chosen.values()
    ]


def truncate(passage: Passage, max_chars: int) -> Passage:
    """The passage with its text cut to at most ``max_chars`` characters.

    The cut is made at the last paragraph break (a blank line, else a line
    break) within the limit, or at the limit when there is none, and is
    marked in the text. A passage without text, or with a text within the
    limit, is returned unchanged.
    """
    text = passage.text
    if text is None or len(text) <= max_chars:
        return passage
    cut = text.rfind("\n\n", 0, max_chars + 1)
    if cut <= 0:
        cut = text.rfind("\n", 0, max_chars + 1)
    if cut <= 0:
        cut = max_chars
    kept = text[:cut].rstrip()
    marker = (
        f"[Text cut by the retriever after {len(kept)} of {len(text)} characters "
        f"(max_chars={max_chars}). The full text is at the source.]"
    )
    return dataclasses.replace(passage, text=f"{kept}\n{marker}")
