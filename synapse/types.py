from __future__ import annotations

from dataclasses import dataclass


class Type:
    pass


@dataclass(frozen=True, slots=True)
class ScalarType(Type):
    name: str


@dataclass(frozen=True, slots=True)
class TensorType(Type):
    dtype: str
    shape: tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class FunctionType(Type):
    params: tuple[Type, ...]
    returns: Type


INT32 = ScalarType("Int32")
INT64 = ScalarType("Int64")
FLOAT32 = ScalarType("Float32")
FLOAT64 = ScalarType("Float64")
BOOL = ScalarType("Bool")
UNKNOWN = ScalarType("Unknown")
