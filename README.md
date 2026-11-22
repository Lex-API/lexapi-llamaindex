# lexapi-llamaindex

LlamaIndex integrations for [LexAPI](https://lex-api.com), an API over EU
law: legislation from EUR-Lex parsed to article level, and CJEU case law.

| Package | Import | Contents |
|---|---|---|
| [`llama-index-retrievers-lexapi`](llama-index-retrievers-lexapi/) | `llama_index.retrievers.lexapi` | `LexAPILegislationRetriever`, `LexAPICaseLawRetriever` |
| [`llama-index-tools-lexapi`](llama-index-tools-lexapi/) | `llama_index.tools.lexapi` | `LexAPIToolSpec` (search, read a document, the version for a date) |

Both packages depend only on `llama-index-core` and the LexAPI Python SDK
(`lexapi-client`), whose `lexapi.passages` module builds every text,
citation and notice.

- Consolidated texts have no legal force: cite the act as published in the
  Official Journal.
- A text is passed on as the text for a date only when LexAPI established it
  for that date.
- LexAPI provides source texts and citations; it does not assess whether
  anyone complies with the law. Not legal advice.

## Install

```bash
pip install llama-index-retrievers-lexapi   # retrievers
pip install llama-index-tools-lexapi        # agent tools
```

Python 3.11 or later, and a LexAPI key in `LEXAPI_API_KEY`.

## Development

Each package is built and tested on its own:

```bash
cd llama-index-retrievers-lexapi   # or llama-index-tools-lexapi
uv venv
. .venv/bin/activate
uv pip install -e . --group dev
ruff check . ../testing ../scripts
ruff format --check . ../testing ../scripts
mypy
pytest
```

The tests are offline: clients answer from `httpx.MockTransport` with the
synthetic fixtures in `testing/fixtures/`, real HTTP transports and sockets
are blocked, and LlamaIndex's LLM and embedding settings point at mock
models. `scripts/mutation_check.py` applies deliberate faults to the source
and checks that the tests catch each one.

## Licence

MIT
