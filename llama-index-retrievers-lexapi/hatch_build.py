"""Build hook: leave the repository's ``.gitignore`` out of the sdist.

Hatchling adds the nearest ``.gitignore`` to every sdist; it is a
development file and not part of the package.
"""

from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):  # type: ignore[type-arg]
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "sdist":
            return
        forced = build_data.get("force_include", {})
        for source in [path for path, target in forced.items() if target == ".gitignore"]:
            del forced[source]
