# llama-index-tools-lexapi

LlamaIndex tools for EU legislation from [LexAPI](https://lex-api.com), for
agents. Every passage is followed by the citation of its source.

| Tool | What it does |
|---|---|
| `lexapi_search_legislation(query, as_of=None)` | Articles of EU legislation by meaning, optionally with the text for a date |
| `lexapi_read_document(celex, article=None, language=None)` | A document by CELEX number: one article, or its articles in order within a character budget; the articles not shown are listed |
| `lexapi_legislation_as_of(celex, as_of, provision=None, language=None)` | The version of an act LexAPI lists for a date, and optionally one provision of it; an answer LexAPI did not establish is always a notice without text |

Each tool has an async variant (`alexapi_...`) used by async agents.

## Install

Version 0.1.0 is not on PyPI yet. Install it from this repository:

```bash
pip install "llama-index-tools-lexapi @ git+https://github.com/Lex-API/lexapi-llamaindex#subdirectory=llama-index-tools-lexapi"
```

After the release it installs from PyPI with
`pip install llama-index-tools-lexapi`.

Python 3.11 or later. You need a LexAPI key (`lex_...`) from your LexAPI
dashboard, in the environment as `LEXAPI_API_KEY` or passed as `api_key=`.

## Usage

In an async context:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.lexapi import LexAPIToolSpec

spec = LexAPIToolSpec()
agent = FunctionAgent(tools=spec.to_tool_list(), llm=my_llm)
response = await agent.run("What did Article 17 of Regulation 2016/679 say on 2019-06-01?")
await spec.aclose()
```

The tools can be called directly as well:

```python
with LexAPIToolSpec() as spec:
    print(spec.lexapi_legislation_as_of("32016R0679", "2019-06-01", provision="art_17"))
```

To use your own SDK clients, pass them in: `LexAPIToolSpec(LexAPI(),
AsyncLexAPI())`. With only a sync client, the async tools run it in a worker
thread; with only an async client, the sync tools raise `ValueError`. A
client you pass in is never closed. Clients the tool spec builds from a key
(the async one on first async use) are closed by `close()` (the sync one) or
`await aclose()` (both), or by leaving a `with` / `async with` block; the
tool spec is then closed and refuses further calls.

Options: `similarity_top_k` (hits per search, 1-50, default 5), `language`,
`min_score`, `unresolved` (below) and `max_chars` (below). Unknown options
raise `TypeError`.

## Output

- **Search:** the passages numbered `[1]`, `[2]`, ... and separated by blank
  lines, each its text or a notice followed by a `Source:` line with the
  citation; or `LexAPI returned no matching articles.` Then, when hits were
  left out, `[N hit(s) left out: LexAPI did not establish their text for
  <date>.]`, then `Hint from LexAPI: ...` when the API sent a hint.
- **Read:** numbered the same way. `max_chars` (at least 200, default
  20000, `None` for no bound) bounds the sum of the article texts and
  notices shown, counted as they are shown; the numbering, `Source:`
  lines, the cut note and the list
  of articles not shown come on top. When the first article alone is over
  the budget its text is cut and ends with `[Text cut by
  llama-index-tools-lexapi: K of N characters shown. The full text is at
  the source.]`; the articles that do not fit are listed as `[Not shown: N
  more article(s) of CELEX <celex>: <refs>. Read one with the article
  argument.]`. A text is cut at a paragraph break when one falls in the
  second half of the limit, otherwise at a line break or a space there,
  otherwise at the limit.
- **As of a date:** the passage, followed by its `Source:` line.

Texts, citations and notices are built by `lexapi.passages` in the LexAPI
Python SDK, so every LexAPI integration shows the same ones. Passages keep
the API's order. Two hits with the same id are merged: the first position
is kept, and a later hit replaces it only when it carries more (a text over
a labelled latest text over a notice).

## The text for a date

- A text is passed on as the text for the date only when LexAPI established
  it for that date.
- Any other search hit is returned as a notice without text
  (`unresolved="withhold"`, the default). With `unresolved="drop"` it is
  left out and the output says how many were left out; with
  `unresolved="keep_labelled"` its latest text is kept, under a label saying
  it is not established for that date. The search tool's description tells
  the model which of these applies.
- `lexapi_legislation_as_of` always withholds, whatever `unresolved` says.
- Dates are checked before any request: a value that is not a real calendar
  day in the form `YYYY-MM-DD` raises `Invalid argument: as_of must be a
  date in the form YYYY-MM-DD; got '<value>'`.

## Credits

A search costs 5 credits; with `as_of`, each version LexAPI does not hold is
fetched (up to 3 per search) at 0-5 credits each. A document read costs 1
credit from LexAPI's corpus, 4 from Cellar and 5 when fetched live.
`lexapi_legislation_as_of` costs 1 credit when no text is fetched, otherwise
4 or 5. See the LexAPI API reference for the current prices.

Every successful call dispatches a `LexAPIToolEvent` instrumentation event
with the call's `LexAPICallReport` and its `lexapi.passages.PassageResult`.
The report's fields: `requested` (hits asked for by a search; `None` for a
read and for the as-of tool), `returned`, `as_of`, the core's counters
`withheld`, `kept_labelled`, `dropped`, `uncitable` and `filtered_out`
(counted before de-duplication), `returned_withheld`, `returned_labelled`,
`withheld_not_emitted` (always 0 for the tools, which show withheld
passages as notices), `duplicates`, `truncated` (texts that were cut),
`not_shown` (articles left out by the read budget), `units_charged`,
`credits_remaining` and `hint`.

## Errors

A failed call raises `LexAPIToolError`. LlamaIndex agents hand its message
to the model as the tool's error output. The model reads this text and may
repeat it to the person it is talking to, including a failed payment and the
billing link; an application that handles billing itself should check
`is_error` and use the attributes (below).

An error the API answered reads `LexAPI error <code> (HTTP <status>):
<message>.` (a period is added unless the message ends in `.`, `!` or `?`),
followed, each only when it applies and in this order, by:

1. ` Credits reset at <time>.`: credits exhausted, when the reset time is
   known.
2. ` This is an exhausted quota, not a rate limit.`: an exhausted quota
   (code `QUOTA_EXHAUSTED`, or a 429 of the legacy quota shape), then
   ` Quota resets at <time>.` when the reset time is known.
3. ` Access ended at <time>.`: a failed payment (`PAYMENT_PAST_DUE`), when
   known.
4. ` Billing: <url>`: a failed payment, when known and not already in the
   message.
5. ` Retry after <n> s.`: only for an error the LexAPI client retries
   under its default retry policy (a rate-limit 429, or a 521, 522 or 523:
   the request never reached LexAPI), when the server named a wait. Never
   on an exhausted quota, a failed payment, exhausted credits or national
   allowance, any other 4xx or any other 5xx.

A failed payment is not shown as a plan gate. Strings the server sent are
put on one line (control characters and line breaks become spaces, runs of
spaces one space) and cut to 200 characters (the code, times, URLs) or
1,000 (the message), followed by `…`; one that is empty after that counts
as not sent. A message ending in `:` ends in `.` instead; a blank one
leaves `LexAPI error <code> (HTTP <status>).`

Other errors:

- An answer that could not be read, or a read or as-of answer of the wrong
  shape: `LexAPI error INVALID_RESPONSE (HTTP <status>): The answer could
  not be read.` (without `(HTTP <status>)` when there is none).
- An invalid argument, before any request: `Invalid argument: ...` (an
  empty `query` or `celex`, a blank `article` or `provision`, a value that
  is not a string, a `language` that is not a two-letter code, a date that
  is not `YYYY-MM-DD`).
- After `close()`: `LexAPIToolSpec is closed; create a new one.`
- SDK exceptions are not attached; an API error's message is the API's own
  text.
- Anything shaped like an API key is masked as `lex_***`.

### Attributes

`LexAPIToolError` carries the billing and reset information as attributes,
so an application can show it without reading the text. An attribute that
does not apply, or that the API did not send, is `None` (`quota_exhausted`
is `False`). Values are the SDK's, unchanged and not cut; a string that is
empty or blank counts as not sent.

| Attribute | Type | Set on |
|---|---|---|
| `code` | `str` | every error: the API's code as sent (`UNKNOWN`, as in the text, when it is blank), or the package's own `INVALID_ARGUMENT`, `CLOSED`, `INVALID_RESPONSE` |
| `status` | `int` or `None` | the HTTP status; `None` without an HTTP answer |
| `retry_after` | `float` or `None` | the wait in seconds the server named, also when a retry cannot help (the text shows it only when it can); `None` on an exhausted quota and a failed payment |
| `quota_exhausted` | `bool` | `True` on an exhausted quota (code `QUOTA_EXHAUSTED`, or a 429 of the legacy quota shape) |
| `resets_at` | `str` or `None` | `CREDITS_EXHAUSTED` and an exhausted quota: when the pool or quota refills (ISO-8601) |
| `operation` | `str` or `None` | `CREDITS_EXHAUSTED`: the refused operation |
| `weight`, `available`, `included`, `topup`, `used` | `int` or `None` | `CREDITS_EXHAUSTED`: credits needed, left, included in the plan, bought as top-ups, used. The API sends no link with this error |
| `upgrade_url` | `str` or `None` | `TIER_FORBIDDEN`: the plan gate's link; may be a path relative to https://lex-api.com (for example `/pricing`). Never set on a failed payment |
| `billing_url` | `str` or `None` | `PAYMENT_PAST_DUE`: where to pay the open invoice |
| `access_ended_at` | `str` or `None` | `PAYMENT_PAST_DUE`: when access ended (ISO-8601) |

An answer that could not be read carries none of them. `LexAPIToolError`
can be pickled and copied with its attributes.

A direct call, and a tool's `call()` / `acall()`, raise `LexAPIToolError`:

```python
from llama_index.tools.lexapi import LexAPIToolError

with LexAPIToolSpec() as spec:
    try:
        spec.lexapi_read_document("32016R0679")
    except LexAPIToolError as err:
        print(err.code, err.resets_at, err.available)

    read = {t.metadata.name: t for t in spec.to_tool_list()}["lexapi_read_document"]
    try:
        read.call(celex="32016R0679")
    except LexAPIToolError as err:
        print("call():", err.code)
```

In an agent, on llama-index-core 0.14 and later, the error is on the tool
output's `exception` when `is_error` is set, both in the `ToolCallResult`
stream event and in the run's `tool_calls`:

```python
# llama-index-core 0.14 and later
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult

spec = LexAPIToolSpec()
agent = FunctionAgent(tools=spec.to_tool_list(), llm=my_llm)
handler = agent.run("What does Article 17 of Regulation 2016/679 say?")
async for event in handler.stream_events():
    if isinstance(event, ToolCallResult) and event.tool_output.is_error:
        print("event:", event.tool_output.exception.code)
response = await handler
for call in response.tool_calls:
    if call.tool_output.is_error and isinstance(call.tool_output.exception, LexAPIToolError):
        print("result:", call.tool_output.exception.code)
await spec.aclose()
```

On any version, and the only way on llama-index-core 0.13 (where
`ToolOutput` has no `exception`), wrap the tools:

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.tools.lexapi import LexAPIToolError, LexAPIToolSpec

spec = LexAPIToolSpec()
errors = []


def wrap(tool):
    async def run(**kwargs):
        try:
            return str((await tool.acall(**kwargs)).raw_output)
        except LexAPIToolError as err:
            errors.append(err)
            raise

    return FunctionTool.from_defaults(async_fn=run, tool_metadata=tool.metadata)


agent = FunctionAgent(tools=[wrap(t) for t in spec.to_tool_list()], llm=my_llm)
await agent.run("What does Article 17 of Regulation 2016/679 say?")
for err in errors:
    print("wrapped:", err.code, err.resets_at)
await spec.aclose()
```

### Other failures

Invalid constructor options (`similarity_top_k`, `unresolved`, `max_chars`,
client wiring) raise `ValueError` when the tool spec is built.

An error is never turned into an empty answer. An answer LexAPI's client
cannot read raises an error; it is never read as an empty result, and a 2xx
answer is never retried. An answer says nothing matched when the search had
no hits, when `unresolved="drop"` left every hit out, or when the API sent
hits without a CELEX number.

## What this package does not do

- LexAPI provides source texts and citations. It does not assess whether
  anyone complies with the law, and neither does this package.
- Consolidated texts are documentation tools without legal force. Cite the
  act as published in the Official Journal (the CELEX number in the
  `Source:` line).
- A text is passed on as the text for a date only when LexAPI established it
  for that date. For any other answer this package passes on no text, unless
  you opt into `unresolved="keep_labelled"` for searches.
- No national law and no case-law search: the tools cover EU legislation.
  Case law is in `llama-index-retrievers-lexapi`.
- Not legal advice.

## Tests

The tests run offline against synthetic fixtures. They do not cover the live
service.

## Licence

MIT
