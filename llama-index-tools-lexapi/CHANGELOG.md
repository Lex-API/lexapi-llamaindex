# Changelog

## 0.1.0 (unreleased)

- `LexAPIToolSpec` with three tools, each with an async variant:
  `lexapi_search_legislation`, `lexapi_read_document` and
  `lexapi_law_on_date`.
- Output built with `lexapi.passages`: every passage ends with its citation.
- Errors raise `LexAPIToolError` with the API's code and message.
- `LexAPIToolEvent` reports each call's credits and counts.
