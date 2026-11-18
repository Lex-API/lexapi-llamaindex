"""De-duplication, cutting and the per-call report for the tools.

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

    - ``requested``: hits asked for by a search; ``None`` for a document
      read and for the version on a date.
    - ``returned``: passages shown in the tool output.
    - ``as_of``: the date asked for.
    - ``withheld``, ``kept_labelled``, ``dropped``, ``uncitable``,
      ``filtered_out``: the counters of :class:`lexapi.passages.PassageResult`,
      counted before de-duplication.
    - ``returned_withheld`` / ``returned_labelled``: returned passages whose
      text was withheld / is a labelled latest text.
    - ``withheld_not_emitted``: always 0 for the tools, which always show
      withheld passages as notices.
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

    Leading whitespace is dropped; a text that then fits is returned as it
    is. Otherwise the cut is made at the last blank line at or before
    ``max_chars``, else the last line break, else the last space, else at
    ``max_chars``, never leaving nothing; a note saying so follows.
    """
    body = text.lstrip()
    if len(body) <= max_chars:
        return body, False
    cut = 0
    for separator in ("\n\n", "\n", " "):
        position = body.rfind(separator, 0, max_chars + len(separator))
        if position > 0:
            cut = position
            break
    if cut <= 0:
        cut = max_chars
    kept = body[:cut].rstrip()
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
