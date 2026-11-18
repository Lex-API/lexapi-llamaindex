# llama-index-tools-lexapi

LlamaIndex tools for EU legislation from [LexAPI](https://lex-api.com), for
agents. Every answer ends with the citation of its source.

| Tool | What it does |
|---|---|
| `lexapi_search_legislation(query, as_of=None)` | Articles of EU legislation by meaning, optionally in the version LexAPI lists for a date |
| `lexapi_read_document(celex, article=None, language=None)` | A document by CELEX number: one article, or its articles up to `max_chars` characters (default 20000; the rest are listed by reference) |
| `lexapi_law_on_date(celex, date, provision=None, language=None)` | The version of an act LexAPI lists for a date, and optionally one provision of it; an answer LexAPI could not establish is always a notice without text |

Each tool has an async variant used by async agents.

## Install

```bash
pip install llama-index-tools-lexapi
```

Python 3.11 or later. You need a LexAPI key (`lex_...`) from your LexAPI
dashboard, in the environment as `LEXAPI_API_KEY` or passed as `api_key=`.

## Usage

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.lexapi import LexAPIToolSpec

tools = LexAPIToolSpec().to_tool_list()
agent = FunctionAgent(tools=tools, llm=my_llm)
response = await agent.run("What did Article 17 of Regulation 2016/679 say on 2019-06-01?")
```

The tools can be called directly as well:

```python
spec = LexAPIToolSpec()
print(spec.lexapi_law_on_date("32016R0679", "2019-06-01", provision="art_17"))
```

To use your own SDK clients, pass them in: `LexAPIToolSpec(LexAPI(),
AsyncLexAPI())`. With only a sync client, the async tools run it in a worker
thread. A client you pass in is never closed by the tool spec. Clients it
builds from a key (the async one on first async use) are closed by
`close()` / `await aclose()`.

Options: `similarity_top_k` (hits per search, 1-50, default 5), `language`,
`min_score`, `unresolved` (below) and `max_chars` (budget of
`lexapi_read_document`, at least 200, default 20000). Unknown options raise
`TypeError`.

## Output

Each tool returns one string: every passage as its text, or a notice, then a
`Source:` line with the citation (built by `lexapi.passages` in the LexAPI
Python SDK, so every LexAPI integration shows the same citation and
notices). A search that matched nothing says so. Two hits with the same id
are merged: the first position is kept, and a later hit replaces it only
when it carries more (a text over a labelled latest text over a notice). A
text cut to fit `max_chars` ends with `[Text cut by llama-index-tools-lexapi:
N of M characters shown. The full text is at the source.]`.

## The text for a date

- A text is passed on as the text for the date only when LexAPI established
  it for that date.
- Any other search hit is returned as a notice with LexAPI's reason and no
  text (`unresolved="withhold"`, the default). With `unresolved="drop"` it
  is left out and counted; with `unresolved="keep_labelled"` the latest
  indexed text is kept, labelled as not established for the date. The
  search tool's description tells the model which of these applies.
- `lexapi_law_on_date` always withholds: an answer LexAPI could not
  establish is a notice without text, whatever `unresolved` says.

## Credits

A search costs 5 credits; with `as_of`, each version LexAPI does not hold is
fetched (up to 3 per request by default) at 0-5 credits each. A document read
costs 1 credit from LexAPI's corpus, 4 from Cellar and 5 when fetched live.
`lexapi_law_on_date` costs 1 credit when no text is fetched, otherwise 4 or 5.
See the LexAPI API reference for the current prices.

Every successful call dispatches a `LexAPIToolEvent` instrumentation event
with the call's `LexAPICallReport` (`requested`, `returned`, `as_of`, the
core's counters `withheld`, `kept_labelled`, `dropped`, `uncitable` and
`filtered_out` counted before de-duplication, `withheld_not_emitted`,
`returned_withheld`, `duplicates`, `truncated`, `units_charged`,
`credits_remaining`, `hint`) and its `lexapi.passages.PassageResult`.

## Errors

A failed call raises `LexAPIToolError` with the API's code, HTTP status and
message, then, when known, when credits reset, the billing page, or when to
retry, for example `LexAPI error CREDITS_EXHAUSTED (HTTP 402): Monthly
credit pool exhausted. Credits reset at 2026-12-01T00:00:00.000Z.` Anything
shaped like an API key is masked. LlamaIndex agents hand that message to
the model as the tool's error output. The SDK's exception is kept as
`__cause__`. An error is never turned into an empty answer. An answer can
say nothing matched when the search had no hits, when `unresolved="drop"`
left every hit out, or when the API sent hits without a CELEX number.

## What this package does not do

- LexAPI provides source texts and citations. It does not assess whether
  anyone complies with the law, and neither does this package.
- Consolidated texts are documentation tools without legal force. Cite the
  act as published in the Official Journal (the CELEX number in the
  `Source:` line).
- A text is passed on as the text for a date only when LexAPI established it
  for that date. For any other answer this package passes on no text, with
  the reason LexAPI gave, unless you opt into `unresolved="keep_labelled"`
  for searches.
- Not legal advice.

## Tests

The tests run offline against synthetic fixtures. They do not cover the live
service.

## Licence

MIT
