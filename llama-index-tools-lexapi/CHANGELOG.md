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
- Read and as-of answers of the wrong shape raise the fixed unreadable-answer
  error. Empty `query` / `celex` and a malformed `language` are refused
  before any request.
- The read budget counts the texts and notices as they are shown.
- Cut rule as in the retrievers package.
- Depends on `lexapi-client` from its repository at commit `19da4dd` until
  0.5.0 is on PyPI (`lexapi-client>=0.5.0,<0.6` before any upload).
- Read answers must carry a non-empty `celex` string and a string
  `content.fullText` when present; the first passage of a read is always
  shown, a notice whole.
