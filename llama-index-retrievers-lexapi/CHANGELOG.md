# Changelog

## [0.1.0] - 2026-11-22

- `LexAPILegislationRetriever`: semantic search over EU legislation, with
  `as_of`, `celex`, `unresolved` (`withhold` / `drop` / `keep_labelled`) and
  `emit_withheld`.
- `LexAPICaseLawRetriever`: semantic search over CJEU case law, always with
  full decision texts, optional `max_chars` (at least 200), and `court` /
  `date_from` / `date_to` filters.
- Nodes built with `lexapi.passages`: the citation in the text's `Source:`
  line, metadata never embedded, and `EXCLUDED_LLM_METADATA_KEYS` (the
  duplicate citation keys) hidden from the LLM view, so the model reads the
  citation once. Hits sharing a node id are merged (first position, best
  text kept).
- Cut rule for `max_chars`: a text within the limit is unchanged; a longer
  one is cut at a paragraph break in the second half of the limit, else a
  line break or a space there, else at the limit, with a note.
- `retrieve_with_result` / `aretrieve_with_result` return the nodes, a
  `LexAPICallReport` and the `PassageResult`; `LexAPIRetrievalEvent`
  carries the same for each retrieval.
- Arguments are checked before any request and refused with
  `InvalidArgumentError` (a `ValueError`): an empty or whitespace-only query
  (`query must not be empty`), and `similarity_top_k`, `max_chars`,
  `unresolved`, `as_of`, `date_from` and `date_to`. A bad `celex`
  restriction and wrong client wiring raise `ValueError`.
- `with` / `async with`, `close()` / `aclose()`; a closed retriever raises
  `ClientClosedError`, before any other check.
- SDK errors propagate unchanged. `lexapi-client` (0.5.0 or later, before 0.6) reads answers
  as strict JSON (`NaN` and `Infinity` are refused) and checks their shape:
  such answers raise `lexapi.InvalidResponseError`.
