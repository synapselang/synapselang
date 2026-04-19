from __future__ import annotations

from dataclasses import dataclass, field


class RuntimeNameError(Exception):
    """Raised when a runtime name lookup fails."""


@dataclass
class Environment:
    parent: "Environment | None" = None
    values: dict[str, object] = field(default_factory=dict)

    def define(self, name: str, value: object) -> None:
        self.values[name] = value

    def assign(self, name: str, value: object) -> None:
        if name in self.values:
            self.values[name] = value
            return
        if self.parent is not None:
            self.parent.assign(name, value)
            return
        raise RuntimeNameError(f"Undefined name: {name}")

    def get(self, name: str) -> object:
        if name in self.values:
            return self.values[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise RuntimeNameError(f"Undefined name: {name}")
