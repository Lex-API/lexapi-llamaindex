# llama-index-retrievers-lexapi

LlamaIndex retrievers for EU legislation and CJEU case law from
[LexAPI](https://lex-api.com), with a citation on every node.

- `LexAPILegislationRetriever`: articles of EU legislation by meaning,
  optionally in the version LexAPI lists for a date (`as_of`).
- `LexAPICaseLawRetriever`: decisions of the Court of Justice of the EU by
  meaning.

## Install

Version 0.1.0 is not on PyPI yet. Install it from this repository:

```bash
pip install "llama-index-retrievers-lexapi @ git+https://github.com/Lex-API/lexapi-llamaindex#subdirectory=llama-index-retrievers-lexapi"
```

After the release it installs from PyPI with `pip install llama-index-retrievers-lexapi`.

Python 3.11 or later. You need a LexAPI key (`lex_...`) from your LexAPI
dashboard, in the environment as `LEXAPI_API_KEY` or passed as `api_key=`.

## Usage

```python
from llama_index.retrievers.lexapi import LexAPILegislationRetriever

retriever = LexAPILegislationRetriever(similarity_top_k=5)
for item in retriever.retrieve("right to erasure of personal data"):
    meta = item.node.metadata
    # cite_celex is absent for a consolidated text whose act LexAPI did not name.
    print(item.score, meta.get("cite_celex"), meta["citation"])
    print(item.node.get_content())
```

Inside a query engine, with the LLM of your choice:

```python
from llama_index.core.query_engine import RetrieverQueryEngine

engine = RetrieverQueryEngine.from_args(retriever, llm=my_llm)
print(engine.query("When must a controller erase personal data?"))
```

To use your own SDK clients (timeouts, retries, base URL), pass them in. In
an async context:

```python
from lexapi import AsyncLexAPI, LexAPI

retriever = LexAPILegislationRetriever(LexAPI(), AsyncLexAPI())
nodes = await retriever.aretrieve("own funds requirements")
```

`retrieve` uses the sync client and `aretrieve` the async one. With only a
sync client, `aretrieve` runs it in a worker thread; with only an async
client, `retrieve` raises `ValueError`. A client you pass in is never
closed. Clients the retriever builds from a key (the async one on first
async use) are closed by `close()` (the sync one) or `await aclose()`
(both), or by leaving a `with` / `async with` block; the retriever is then
closed and refuses further calls.

## Nodes

Each hit becomes one `TextNode` in a `NodeWithScore`:

| Node field | Value |
|---|---|
| `text` | The text, then a `Source:` line with the citation. With `include_citation_in_text=False`, the text alone. |
| `id_` | `lexapi:<celex>:<part>:<as_of>:<language>`, stable across calls |
| `metadata` | The passage's citation fields (below), never the text |
| `score` | The API's similarity score, unchanged |

The text and metadata are built by `lexapi.passages` in the LexAPI Python
SDK, so every LexAPI integration shows the same citation and notices.

Metadata keys (present when known): `kind`, `celex`, `cite_celex`,
`citation`, `id`, `text_status`, `text_withheld`, `consolidated`,
`base_celex`, `consolidated_as_of`, `consolidated_note`, `article_ref`,
`article_number`, `article_title`, `title`, `language`, `url`, `eli`, `score`,
`as_of`, `as_of_resolved`, `as_of_reason`, `version_celex`, `version_kind`,
`version_date`, `version_valid_until`, and for case law `ecli`,
`case_number`, `court`, `decision_date`, `document_type`,
`paragraph_number`.

- **Embedding:** no metadata key is embedded
  (`excluded_embed_metadata_keys` lists them all).
- **LLM view:** the citation is read once, in the `Source:` line; the keys
  the model sees are listed below.
  - Hidden (`EXCLUDED_LLM_METADATA_KEYS`): `base_celex`, `celex`,
    `citation`, `eli`, `id`, `kind`, `score`, `text_withheld`, `url`,
    `version_celex`, `version_date`, `version_kind`, `version_valid_until`.
    With `include_citation_in_text=False` there is no `Source:` line and
    `citation` is shown instead.
  - Seen: every other key that is present, among them `cite_celex`,
    `consolidated`, `consolidated_note`, `as_of`, `as_of_resolved`,
    `as_of_reason`, `text_status`, `title`, `article_ref`, `language` and,
    for case law, `ecli`, `court`, `decision_date`.

## The text for a date

```python
retriever = LexAPILegislationRetriever(as_of="2019-06-01")
```

The date must be a real calendar day in the form `YYYY-MM-DD` (or a
`datetime.date`); anything else raises `ValueError` before any request.

- A text is passed on as the text for the date only when LexAPI established
  it for that date.
- Any other hit is kept as a node whose text is a notice with LexAPI's
  reason and no text (`unresolved="withhold"`, the default). With
  `unresolved="drop"` it is left out; with `unresolved="keep_labelled"` the
  latest indexed text is kept, labelled as not established for the date.
- Withheld nodes have `metadata["text_withheld"] == True`. Do not index or
  embed them; `emit_withheld=False` leaves them out of the results.
- The date is set per retriever. LlamaIndex's `retrieve()` takes only the
  query, so for another date build another retriever.

## Restricting to one act

`celex="32016R0679"` keeps only hits of that act and its consolidated texts.
The search itself cannot be limited to an act: the retriever filters the
`similarity_top_k` hits it asked for, so the result can be shorter or empty.
It never asks for more hits to make up the number.

## Case law

`LexAPICaseLawRetriever` always asks for the full text of each decision
(`include_text=True`). The text of a node is the whole decision, which can
be long. With `max_chars=N` (at least 200) a longer text is cut at a
paragraph break when one falls in the second half of the limit, otherwise
at a line break or a space there, otherwise at the limit, and the cut is
marked with `[Text cut by llama-index-retrievers-lexapi: K of M characters
shown. The full text is at the source.]` before the `Source:` line. The
default is the whole decision. The search snippet and
the machine-written summary are never used as text. Filters: `court`,
`date_from`, `date_to`.

## Credits

`similarity_top_k` is the number of hits requested. A legislation search
costs 5 credits; with `as_of`, each version LexAPI does not hold is fetched
(up to 3 per search) at 0-5 credits each. A case-law search
costs 5 credits. See the LexAPI API reference for the current prices.

What a call did is reported per call, never stored on the retriever:

```python
nodes, report, result = retriever.retrieve_with_result("right to erasure")
print(report.requested, report.returned, report.units_charged, report.credits_remaining)
```

`report` is a `LexAPICallReport`: `requested`, `returned`, `as_of`, the
core's counters `withheld`, `kept_labelled`, `dropped`, `uncitable` and
`filtered_out` (counted before de-duplication), `returned_withheld`,
`returned_labelled`, `withheld_not_emitted`, `duplicates`, `truncated`
(returned texts that were cut), `not_shown` (always 0 for the retrievers),
`units_charged`, `credits_remaining` and `hint`. `result` is the call's
`lexapi.passages.PassageResult`. Inside a query engine, subscribe an
instrumentation event handler to `LexAPIRetrievalEvent`, which carries both
for every retrieval.

Nodes keep the API's order. Two hits with the same node id are merged: the
first position is kept, and a later hit replaces the earlier one only when
it carries more (a text over a labelled latest text over a notice). Each
node keeps its own score, so after a merge the scores need not be
descending.

## Errors

An error is never turned into an empty list. An answer LexAPI's client
cannot read raises an error; it is never read as an empty result, and a 2xx
answer is never retried. A result
can be empty when the search had no hits, when `celex` filtered every hit out, when
`unresolved="drop"` left every hit out, when `emit_withheld=False` left out
withheld hits, or when the API sent hits without a CELEX number (counted in
`report.uncitable`). SDK exceptions are raised unchanged; an API error's
message is the API's own text. For example: `lexapi.CreditsExhaustedError`
(402), `lexapi.PaymentPastDueError` (403), `lexapi.RateLimitedError` (429),
`lexapi.AuthenticationError` (401), and `lexapi.InvalidResponseError` for
an answer that could not be read. Using a retriever after `close()` raises
`ClientClosedError`.

## What this package does not do

- LexAPI provides source texts and citations. It does not assess whether
  anyone complies with the law, and neither does this package.
- Consolidated texts are documentation tools without legal force. Cite the
  act as published in the Official Journal (`cite_celex`).
- A text is passed on as the text for a date only when LexAPI established it
  for that date. For any other hit this package passes on no text, with the
  reason LexAPI gave, unless you opt into `unresolved="keep_labelled"`,
  which passes on the latest text labelled as not established for the date.
- Not legal advice. A retriever sees only what the search returned.

## Tests

The tests run offline against synthetic fixtures, with mock LLM and
embedding models. They do not cover the live service.

## Licence

MIT
