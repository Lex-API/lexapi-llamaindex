# Changelog

## 0.1.0 (unreleased)

- `LexAPILegislationRetriever`: semantic search over EU legislation, with
  `as_of`, `celex`, `unresolved` (`withhold` / `drop` / `keep_labelled`) and
  `emit_withheld`.
- `LexAPICaseLawRetriever`: semantic search over CJEU case law, with full
  decision texts, optional `max_chars`, and `court` / `date_from` / `date_to`
  filters.
- Nodes built with `lexapi.passages`: citation in the text, metadata never
  embedded, `EXCLUDED_LLM_METADATA_KEYS` hidden from the LLM view.
- `retrieve_with_result` / `aretrieve_with_result` and
  `LexAPIRetrievalEvent` report each call's credits and counts.
