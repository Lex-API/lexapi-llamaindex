# Changelog

## 0.1.0 (unreleased)

- `LexAPIToolSpec` with three tools, each with an async variant:
  `lexapi_search_legislation`, `lexapi_read_document` and
  `lexapi_legislation_as_of`.
- Output built with `lexapi.passages`: numbered passages, each followed by
  its citation; then the left-out note and the API's hint. The search
  tool's description follows the `unresolved` policy.
- `lexapi_read_document` bounds the sum of the texts and notices shown (as
  they are shown) by `max_chars` (default 20000, `None` for no bound) and
  lists the articles not shown; the first passage is always shown, a text
  cut by the same rule as in the retrievers package, a notice whole.
- `lexapi_legislation_as_of` always withholds an answer LexAPI did not
  establish.
- Arguments are checked before any request with `Invalid argument: ...`: an
  empty `query` or `celex`, a blank `article` (read tool) or `provision`
  (as-of tool) (`... must not be empty`; an absent one is "not given"), a
  value that is not a string (`... must be a string`), a malformed
  `language` and dates.
- Errors raise `LexAPIToolError` with the API's code and message, the credit
  reset time, billing page or retry wait when known; key-shaped text is
  masked. An unreadable answer gets a fixed text and the SDK error is not
  attached: `LexAPI error INVALID_RESPONSE (HTTP 200): The answer could not
  be read.` for answers the pinned `lexapi-client` refuses (wrong shape, or
  not strict JSON: `NaN` and `Infinity` are refused), and the text without
  the status for the tools' own checks (a read answer must carry a
  non-empty `celex` string and a string `content.fullText` when present; an
  as-of `document` must not be blank).
- `LexAPIToolEvent` carries each call's `LexAPICallReport`.
- `with` / `async with`, `close()` / `aclose()`.
- Depends on `lexapi-client` from its repository at commit `19da4dd` until
  0.5.0 is on PyPI (`lexapi-client>=0.5.0,<0.6` before any upload).
