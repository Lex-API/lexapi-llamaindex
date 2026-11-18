# Changelog

## 0.1.0 (unreleased)

- `LexAPIToolSpec` with three tools, each with an async variant:
  `lexapi_search_legislation`, `lexapi_read_document` and
  `lexapi_legislation_as_of`.
- Output built with `lexapi.passages`: numbered passages, each followed by
  its citation; then the left-out note and the API's hint.
- `lexapi_read_document` bounds the sum of the texts shown by `max_chars`
  (default 20000, `None` for no bound) and lists the articles not shown.
  `lexapi_legislation_as_of` always withholds an answer LexAPI did not
  establish.
- Dates are checked before any request. The search tool's description
  follows the `unresolved` policy.
- Errors raise `LexAPIToolError` with the API's code and message, the credit
  reset time, billing page or retry wait when known; an unreadable answer
  gets a fixed text and the SDK error is not attached; key-shaped text is
  masked.
- `LexAPIToolEvent` carries each call's `LexAPICallReport`.
- `with` / `async with`, `close()` / `aclose()`.
