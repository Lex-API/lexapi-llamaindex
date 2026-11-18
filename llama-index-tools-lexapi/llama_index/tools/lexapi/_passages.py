"""De-duplication, truncation and the per-call report for the tools.

Kept in step with ``llama_index.retrievers.lexapi._nodes`` (the two
distributions do not depend on each other).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass

from lexapi.passages import Passage

DISTRIBUTION = "llama-index-tools-lexapi"

#: Smallest accepted ``max_chars``.
MIN_MAX_CHARS = 200


@dataclass(frozen=True)
class LexAPICallReport:
    """What one call did.

    ``requested`` is the number of hits asked for and ``returned`` the number
    of passages shown. ``withheld``, ``kept_labelled``, ``dropped``,
    ``uncitable`` and ``filtered_out`` are the counters of
    :class:`lexapi.passages.PassageResult`, counted before de-duplication.
    ``withheld_not_emitted`` counts withheld passages left out by
    ``emit_withheld=False``; ``returned_withheld`` the withheld passages
    shown; ``duplicates`` the passages merged into another of the same
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
