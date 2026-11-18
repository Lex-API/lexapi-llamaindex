"""Client resolution shared by the tool spec.

No client is created at import and none is shared between instances. A
client the caller passes in is used as it is and never closed here. When
the caller passes a key (or relies on ``LEXAPI_API_KEY``) instead, the
tool spec builds its own sync and async clients at construction, so the key
is handed to the SDK once and not kept by this package.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from lexapi import AsyncLexAPI, LexAPI

DISTRIBUTION = "llama-index-tools-lexapi"


def user_agent_suffix() -> str:
    """``llama-index-tools-lexapi/<version>``, added to the SDK's
    User-Agent on clients this package builds."""
    try:
        installed = version(DISTRIBUTION)
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        installed = "0+unknown"
    return f"{DISTRIBUTION}/{installed}"


class Clients:
    """The sync and async clients one retriever uses."""

    __slots__ = ("_async", "_owned", "_sync")

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
        if client is not None or async_client is not None:
            if api_key is not None or base_url is not None:
                raise ValueError(
                    "Pass either client/async_client or api_key/base_url, not both: "
                    "configure the key and base URL on the client you pass."
                )
            self._sync = client
            self._async = async_client
            self._owned = False
            return
        kwargs: dict[str, Any] = {"user_agent_suffix": user_agent_suffix()}
        if base_url is not None:
            kwargs["base_url"] = base_url
        # The SDK reads LEXAPI_API_KEY when api_key is None, and raises
        # ValueError when neither is set.
        self._sync = LexAPI(api_key, **kwargs)
        self._async = AsyncLexAPI(api_key, **kwargs)
        self._owned = True

    @property
    def sync(self) -> LexAPI:
        if self._sync is None:
            raise TypeError(
                "This tool spec was given only an async_client; use the async tools, "
                "or pass a lexapi.LexAPI as client as well."
            )
        return self._sync

    @property
    def async_(self) -> AsyncLexAPI | None:
        """The async client, or ``None`` when only a sync client was given
        (async calls then run the sync client in a worker thread)."""
        return self._async

    def close(self) -> None:
        """Close the clients this package built; a client passed in is left
        open."""
        if self._owned and self._sync is not None:
            self._sync.close()

    async def aclose(self) -> None:
        if self._owned and self._async is not None:
            await self._async.close()

    def __repr__(self) -> str:
        return f"Clients(owned={self._owned})"
