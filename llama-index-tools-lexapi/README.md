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

After the release it installs from PyPI with `pip install llama-index-tools-lexapi`.

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
  notices shown, counted as they are shown; the numbering, `Source:` lines, the cut note and the list
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
to the model as the tool's error output.

- An error the API answered: `LexAPI error <code> (HTTP <status>):
  <message>.`, then, when known, ` Credits reset at <time>.`,
  ` Billing: <url>` and ` Retry after <n> s.`
- An answer that could not be read, or a read or as-of answer of the wrong
  shape: `LexAPI error INVALID_RESPONSE (HTTP <status>): The answer could
  not be read.` (without `(HTTP <status>)` when there is none).
- An invalid argument, before any request: `Invalid argument: ...` (an
  empty `query` or `celex`, a `language` that is not a two-letter code, a
  date that is not `YYYY-MM-DD`).
- After `close()`: `LexAPIToolSpec is closed; create a new one.`
- SDK exceptions are not attached; an API error's message is the API's own
  text.
- Anything shaped like an API key is masked as `lex_***`.

An error is never turned into an empty answer. An answer LexAPI's client
cannot read raises an error; it is never read as an empty result, and a 2xx
answer is never retried. An answer
says nothing matched when the search had no hits, when `unresolved="drop"`
left every hit out, or when the API sent hits without a CELEX number.

## What this package does not do

- LexAPI provides source texts and citations. It does not assess whether
  anyone complies with the law, and neither does this package.
- Consolidated texts are documentation tools without legal force. Cite the
  act as published in the Official Journal (the CELEX number in the
  `Source:` line).
- A text is passed on as the text for a date only when LexAPI established it
  for that date. For any other answer this package passes on no text, unless
  you opt into `unresolved="keep_labelled"` for searches.
- Not legal advice.

## Tests

The tests run offline against synthetic fixtures. They do not cover the live
service.

## Licence

MIT
