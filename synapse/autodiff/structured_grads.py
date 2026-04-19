from __future__ import annotations

from dataclasses import fields, is_dataclass, make_dataclass
from typing import Any

from .tensor import Tensor


def extract_tensor_grads(value: Any) -> Any:
    if isinstance(value, Tensor):
        return None if value.grad is None else Tensor.from_value(value.grad)
    if isinstance(value, tuple):
        return tuple(extract_tensor_grads(v) for v in value)
    if is_dataclass(value):
        grad_fields = []
        values = []
        for f in fields(value):
            grad_fields.append((f.name, object))
            values.append(extract_tensor_grads(getattr(value, f.name)))
        GradType = make_dataclass(type(value).__name__ + "Grads", grad_fields)
        return GradType(*values)
    return None
