from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from synapse.autodiff.tensor import Tensor


@dataclass(slots=True)
class UserFunction:
    name: str
    params: list[str]
    body: object
    closure: object


@dataclass(slots=True)
class BuiltinFunction:
    name: str
    impl: Callable[..., object]

    def __call__(self, *args: object) -> object:
        return self.impl(*args)


@dataclass(slots=True)
class StructType:
    name: str
    fields: list[str]


__all__ = ["Tensor", "UserFunction", "BuiltinFunction", "StructType"]
