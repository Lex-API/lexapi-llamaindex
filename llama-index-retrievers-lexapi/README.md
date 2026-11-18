# llama-index-retrievers-lexapi

LlamaIndex retrievers for EU legislation and CJEU case law from
[LexAPI](https://lex-api.com), with a citation on every node.

- `LexAPILegislationRetriever`: articles of EU legislation by meaning,
  optionally in the version LexAPI lists for a date (`as_of`).
- `LexAPICaseLawRetriever`: decisions of the Court of Justice of the EU by
  meaning.

## Install

```bash
pip install llama-index-retrievers-lexapi
```

Python 3.11 or later. You need a LexAPI key (`lex_...`) from your LexAPI
dashboard, in the environment as `LEXAPI_API_KEY` or passed as `api_key=`.

## Usage

```python
from llama_index.retrievers.lexapi import LexAPILegislationRetriever

retriever = LexAPILegislationRetriever(similarity_top_k=5)
for item in retriever.retrieve("right to erasure of personal data"):
    print(item.score, item.node.metadata["citation"])
    print(item.node.get_content())
```

Inside a query engine, with the LLM of your choice:

```python
from llama_index.core.query_engine import RetrieverQueryEngine

engine = RetrieverQueryEngine.from_args(retriever, llm=my_llm)
print(engine.query("When must a controller erase personal data?"))
```

To use your own SDK clients (timeouts, retries, base URL), pass them in:

```python
from lexapi import AsyncLexAPI, LexAPI

retriever = LexAPILegislationRetriever(LexAPI(), AsyncLexAPI())
nodes = await retriever.aretrieve("own funds requirements")
```

`retrieve` uses the sync client and `aretrieve` the async one. With only a
sync client, `aretrieve` runs it in a worker thread. A client you pass in is
never closed by the retriever; clients it builds from a key are closed by
`close()` / `await aclose()`.

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
- **LLM view:** `eli`, `id`, `kind`, `score` and `url` are hidden from the
  LLM (`EXCLUDED_LLM_METADATA_KEYS`; the URL is already in `citation`).
  Every other key, including `citation`, `cite_celex`, `consolidated`,
  `as_of`, `as_of_resolved`, `as_of_reason` and `text_status`, is shown.

## The text for a date

```python
retriever = LexAPILegislationRetriever(as_of="2019-06-01")
```

- A text is passed on as the text for the date only when LexAPI established
  it for that date.
- Any other hit is kept as a node whose text is a notice with LexAPI's
  reason and no text (`unresolved="withhold"`, the default). With
  `unresolved="drop"` it is left out; with `unresolved="keep_labelled"` the
  latest indexed text is kept, labelled as not established for the date.
- Withheld nodes have `metadata["text_withheld"] == True`. Do not index or
  embed them; `emit_withheld=False` leaves them out of the results.

## Restricting to one act

`celex="32016R0679"` keeps only hits of that act and its consolidated texts.
The search itself cannot be limited to an act: the retriever filters the
`similarity_top_k` hits it asked for, so the result can be shorter or empty.
It never asks for more hits to make up the number.

## Case law

`LexAPICaseLawRetriever` asks for the full text of each decision
(`include_text=True`). The text of a node is the whole decision, which can
be long; `max_chars=N` cuts it at a paragraph break within `N` characters
and marks the cut. The default is the whole decision. The search snippet and
the machine-written summary are never used as text. Filters: `court`,
`date_from`, `date_to`.

## Credits

`similarity_top_k` is the number of hits requested. A legislation search
costs 5 credits; with `as_of`, each version LexAPI does not hold is fetched
(up to 3 per request by default) at 0-5 credits each. A case-law search
costs 5 credits. See the LexAPI API reference for the current prices.

What a call cost is reported per call, never stored on the retriever:

```python
nodes, result = retriever.retrieve_with_result("right to erasure")
print(result.units_charged, result.credits_remaining, result.withheld)
```

`result` is the call's `lexapi.passages.PassageResult`. Inside a query
engine, subscribe an instrumentation event handler to
`LexAPIRetrievalEvent`, which carries the same result for every retrieval.

## Errors

Errors are raised, never turned into an empty result: an empty list means
the search returned no hits. The SDK's exceptions propagate unchanged, for
example `lexapi.CreditsExhaustedError` (402), `lexapi.PaymentPastDueError`
(403), `lexapi.RateLimitedError` (429) and `lexapi.AuthenticationError`
(401).

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
