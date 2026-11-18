# Changelog

## 0.1.0 (unreleased)

- `LexAPILegislationRetriever`: semantic search over EU legislation, with
  `as_of`, `celex`, `unresolved` (`withhold` / `drop` / `keep_labelled`) and
  `emit_withheld`.
- `LexAPICaseLawRetriever`: semantic search over CJEU case law, always with
  full decision texts, optional `max_chars` (at least 200), and `court` /
  `date_from` / `date_to` filters.
- Hits sharing a node id are merged (first position, best text kept).
- Nodes built with `lexapi.passages`: citation in the text, metadata never
  embedded, `EXCLUDED_LLM_METADATA_KEYS` hidden from the LLM view.
- `retrieve_with_result` / `aretrieve_with_result` return the nodes, a
  `LexAPICallReport` and the `PassageResult`; `LexAPIRetrievalEvent`
  carries the same for each retrieval.
- `as_of`, `date_from` and `date_to` are checked before any request.
- `with` / `async with`, `close()` / `aclose()`; a closed retriever refuses
  calls.
- `ClientClosedError` for use after `close()` / `aclose()`.
- LLM view hides the duplicate citation keys; the model reads the citation
  once, in the `Source:` line.
- Cut rule: a text within the limit is unchanged; a longer one is cut at a
  paragraph break in the second half of the limit, else a line break or a
  space there, else at the limit.
- Depends on `lexapi-client` from its repository at commit `19da4dd` until
  0.5.0 is on PyPI (`lexapi-client>=0.5.0,<0.6` before any upload).
