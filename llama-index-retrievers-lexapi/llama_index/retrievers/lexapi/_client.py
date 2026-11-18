"""Client resolution.

No client is created at import and none is shared between instances. A
client the caller passes in is used as it is and never closed here. When
the caller passes a key (or relies on ``LEXAPI_API_KEY``) instead, the sync
client is built at construction and the async client on first async use;
both can be closed with ``close()`` / ``aclose()``. The key is held only in
a wrapper whose ``repr`` is masked.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from lexapi import AsyncLexAPI, LexAPI

DISTRIBUTION = "llama-index-retrievers-lexapi"


def user_agent_suffix() -> str:
    """``<distribution>/<version>``, added to the SDK's User-Agent on
    clients this package builds."""
    try:
        installed = version(DISTRIBUTION)
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        installed = "0+unknown"
    return f"{DISTRIBUTION}/{installed}"


class _Secret:
    """Holds the API key; never shows it."""

    __slots__ = ("_value",)

    def __init__(self, value: str | None) -> None:
        self._value = value

    def get(self) -> str | None:
        return self._value

    def __repr__(self) -> str:
        return "_Secret(***)"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("the API key cannot be pickled")


class Clients:
    """The sync and async clients one instance uses."""

    __slots__ = ("_async", "_key", "_kwargs", "_owned", "_sync")

    def __init__(
        self,
        client: LexAPI | None,
        async_client: AsyncLexAPI | None,
        api_key: str | None,
        base_url: str | None,
    ) -> None:
        if client is not None and not isinstance(client, LexAPI):
            raise TypeError("client must be a lexapi.LexAPI")
        if async_client is not None and not isinstance(async_client, AsyncLexAPI):
            raise TypeError("async_client must be a lexapi.AsyncLexAPI")
        self._key = _Secret(None)
        self._kwargs: dict[str, Any] = {}
        if client is not None or async_client is not None:
            if api_key is not None or base_url is not None:
                raise ValueError(
                    "Pass either client/async_client or api_key/base_url, not both: "
                    "configure the key and base URL on the client you pass."
                )
            self._sync: LexAPI | None = client
            self._async: AsyncLexAPI | None = async_client
            self._owned = False
            return
        self._kwargs = {"user_agent_suffix": user_agent_suffix()}
        if base_url is not None:
            self._kwargs["base_url"] = base_url
        # The SDK reads LEXAPI_API_KEY when api_key is None, and raises
        # ValueError when neither is set.
        self._sync = LexAPI(api_key, **self._kwargs)
        self._key = _Secret(api_key)
        self._async = None
        self._owned = True

    @property
    def sync(self) -> LexAPI:
        if self._sync is None:
            raise TypeError(
                "This instance was given only an async_client; use the async methods, "
                "or pass a lexapi.LexAPI as client as well."
            )
        return self._sync

    @property
    def async_(self) -> AsyncLexAPI | None:
        """The async client, built on first use when this instance owns its
        clients; ``None`` when only a sync client was passed in (async calls
        then run the sync client in a worker thread)."""
        if self._async is None and self._owned:
            self._async = AsyncLexAPI(self._key.get(), **self._kwargs)
        return self._async

    def close(self) -> None:
        """Close the sync client this package built; a client passed in is
        left open."""
        if self._owned and self._sync is not None:
            self._sync.close()

    async def aclose(self) -> None:
        """Close the async client this package built, if it was built."""
        if self._owned and self._async is not None:
            await self._async.close()

    def __repr__(self) -> str:
        return f"Clients(owned={self._owned})"
