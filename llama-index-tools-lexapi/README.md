# llama-index-tools-lexapi

LlamaIndex tools for EU legislation from [LexAPI](https://lex-api.com), for
agents. Every answer ends with the citation of its source.

| Tool | What it does |
|---|---|
| `lexapi_search_legislation(query, as_of=None)` | Articles of EU legislation by meaning, optionally in the version LexAPI lists for a date |
| `lexapi_read_document(celex, article=None, language=None)` | A document by CELEX number: one article, or every article |
| `lexapi_law_on_date(celex, date, provision=None, language=None)` | The version of an act LexAPI lists for a date, and optionally one provision of it |

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
thread. A client you pass in is never closed by the tool spec.

Options: `similarity_top_k` (hits per search, 1-50, default 5), `language`,
`min_score`, and `unresolved` (below).

## Output

Each tool returns one string: every passage as its text, or a notice, then a
`Source:` line with the citation (built by `lexapi.passages` in the LexAPI
Python SDK, so every LexAPI integration shows the same citation and
notices). A search that matched nothing says so.

## The text for a date

- A text is passed on as the text for the date only when LexAPI established
  it for that date.
- Any other hit is returned as a notice with LexAPI's reason and no text
  (`unresolved="withhold"`, the default). With `unresolved="drop"` it is
  left out and counted; with `unresolved="keep_labelled"` the latest
  indexed text of a search hit is kept, labelled as not established for the
  date.

## Credits

A search costs 5 credits; with `as_of`, each version LexAPI does not hold is
fetched (up to 3 per request by default) at 0-5 credits each. A document read
costs 1 credit from LexAPI's corpus, 4 from Cellar and 5 when fetched live.
`lexapi_law_on_date` costs 1 credit when no text is fetched, otherwise 4 or 5.
See the LexAPI API reference for the current prices.

Every successful call dispatches a `LexAPIToolEvent` instrumentation event
with the call's `lexapi.passages.PassageResult` (`units_charged`,
`credits_remaining`, withheld and dropped counts).

## Errors

A failed call raises `LexAPIToolError` with the API's code, HTTP status and
message, for example `LexAPI error CREDITS_EXHAUSTED (HTTP 402): Monthly
credit pool exhausted`. LlamaIndex agents hand that message to the model as
the tool's error output. The SDK's exception is kept as `__cause__`. Errors
are never turned into an empty answer.

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
