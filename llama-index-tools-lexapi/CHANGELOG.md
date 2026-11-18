# Changelog

## 0.1.0 (unreleased)

- `LexAPIToolSpec` with three tools, each with an async variant:
  `lexapi_search_legislation`, `lexapi_read_document` and
  `lexapi_law_on_date`.
- Output built with `lexapi.passages`: every passage ends with its citation.
- `lexapi_read_document` output is bounded by `max_chars` (default 20000);
  articles beyond it are listed by reference. `lexapi_law_on_date` always
  withholds an answer LexAPI could not establish.
- The search tool's description follows the `unresolved` policy.
- Errors raise `LexAPIToolError` with the API's code and message, the credit
  reset time, billing page or retry wait when known; key-shaped text masked.
- `LexAPIToolEvent` carries each call's `LexAPICallReport`.
