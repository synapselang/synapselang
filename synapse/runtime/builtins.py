from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Any, Callable

from synapse.autodiff.structured_grads import extract_tensor_grads
from synapse.autodiff.tensor import Tensor, detach, ensure_tensor, mean, sum_tensor
from synapse.runtime.values import BuiltinFunction


class GradFunction:
    def __init__(self, fn: Callable[..., object]) -> None:
        self.fn = fn

    def __call__(self, *args: object) -> object:
        prepared = [self._mark_trainable(arg) for arg in args]
        result = self.fn(*prepared)
        if not isinstance(result, Tensor):
            raise TypeError("grad requires a Tensor-valued function in the v0 skeleton")
        result.backward()
        if len(prepared) == 1:
            return extract_tensor_grads(prepared[0])
        return tuple(extract_tensor_grads(arg) for arg in prepared)

    def _mark_trainable(self, value: object) -> object:
        if isinstance(value, Tensor):
            return Tensor.from_value(value.data, requires_grad=True)
        if isinstance(value, (int, float, list, tuple)):
            return Tensor.from_value(value, requires_grad=True)
        if is_dataclass(value):
            kwargs = {}
            for name in value.__dataclass_fields__:
                kwargs[name] = self._mark_trainable(getattr(value, name))
            return type(value)(**kwargs)
        return value


class ValueAndGradFunction(GradFunction):
    def __call__(self, *args: object) -> tuple[object, object]:
        prepared = [self._mark_trainable(arg) for arg in args]
        result = self.fn(*prepared)
        if not isinstance(result, Tensor):
            raise TypeError("value_and_grad requires a Tensor-valued function in the v0 skeleton")
        result.backward()
        grads = extract_tensor_grads(prepared[0]) if len(prepared) == 1 else tuple(extract_tensor_grads(a) for a in prepared)
        return result, grads


def builtin_print(*args: object) -> None:
    rendered = []
    for arg in args:
        if isinstance(arg, Tensor):
            rendered.append(str(arg.data))
        else:
            rendered.append(str(arg))
    print(*rendered)


def builtin_tensor(value: Any) -> Tensor:
    return Tensor.from_value(value)


def builtin_mean(value: object) -> Tensor:
    return mean(ensure_tensor(value))


def builtin_sum(value: object) -> Tensor:
    return sum_tensor(ensure_tensor(value))


def make_builtins() -> dict[str, BuiltinFunction]:
    return {
        "print": BuiltinFunction("print", builtin_print),
        "tensor": BuiltinFunction("tensor", builtin_tensor),
        "mean": BuiltinFunction("mean", builtin_mean),
        "sum": BuiltinFunction("sum", builtin_sum),
        "detach": BuiltinFunction("detach", lambda x: detach(ensure_tensor(x))),
        "grad": BuiltinFunction("grad", lambda fn: GradFunction(fn)),
        "value_and_grad": BuiltinFunction("value_and_grad", lambda fn: ValueAndGradFunction(fn)),
    }
