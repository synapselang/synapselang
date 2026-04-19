from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

BackwardFn = Callable[[np.ndarray], None]


@dataclass
class Tensor:
    data: np.ndarray
    requires_grad: bool = False
    grad: np.ndarray | None = None
    parents: list["Tensor"] = field(default_factory=list)
    backward_fn: BackwardFn | None = None
    op: str | None = None

    @classmethod
    def from_value(cls, value: object, requires_grad: bool = False) -> "Tensor":
        return cls(data=np.array(value, dtype=float), requires_grad=requires_grad)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=float)

    def backward(self, grad: np.ndarray | None = None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward requires an explicit gradient for non-scalar tensors")
            grad = np.ones_like(self.data, dtype=float)
        topo: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                visit(parent)
            topo.append(node)

        visit(self)
        self.grad = grad
        for node in reversed(topo):
            if node.backward_fn is not None and node.grad is not None:
                node.backward_fn(node.grad)

    def __add__(self, other: object) -> "Tensor":
        return add(self, ensure_tensor(other))

    def __radd__(self, other: object) -> "Tensor":
        return add(ensure_tensor(other), self)

    def __sub__(self, other: object) -> "Tensor":
        return sub(self, ensure_tensor(other))

    def __rsub__(self, other: object) -> "Tensor":
        return sub(ensure_tensor(other), self)

    def __mul__(self, other: object) -> "Tensor":
        return mul(self, ensure_tensor(other))

    def __rmul__(self, other: object) -> "Tensor":
        return mul(ensure_tensor(other), self)

    def __truediv__(self, other: object) -> "Tensor":
        return div(self, ensure_tensor(other))

    def __matmul__(self, other: object) -> "Tensor":
        return matmul(self, ensure_tensor(other))

    def __pow__(self, power: object) -> "Tensor":
        return pow_tensor(self, ensure_tensor(power))

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data}, grad={self.grad})"


def ensure_tensor(value: object) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return Tensor.from_value(value)


def _accumulate_grad(tensor: Tensor, grad: np.ndarray) -> None:
    if tensor.grad is None:
        tensor.grad = np.array(grad, dtype=float)
    else:
        tensor.grad = tensor.grad + grad


def _reduce_broadcasted(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad, parents=[a, b], op="add")

    def backward_fn(grad: np.ndarray) -> None:
        if a.requires_grad:
            _accumulate_grad(a, _reduce_broadcasted(grad, a.shape))
        if b.requires_grad:
            _accumulate_grad(b, _reduce_broadcasted(grad, b.shape))

    out.backward_fn = backward_fn if out.requires_grad else None
    return out


def sub(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad, parents=[a, b], op="sub")

    def backward_fn(grad: np.ndarray) -> None:
        if a.requires_grad:
            _accumulate_grad(a, _reduce_broadcasted(grad, a.shape))
        if b.requires_grad:
            _accumulate_grad(b, _reduce_broadcasted(-grad, b.shape))

    out.backward_fn = backward_fn if out.requires_grad else None
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad, parents=[a, b], op="mul")

    def backward_fn(grad: np.ndarray) -> None:
        if a.requires_grad:
            _accumulate_grad(a, _reduce_broadcasted(grad * b.data, a.shape))
        if b.requires_grad:
            _accumulate_grad(b, _reduce_broadcasted(grad * a.data, b.shape))

    out.backward_fn = backward_fn if out.requires_grad else None
    return out


def div(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data / b.data, requires_grad=a.requires_grad or b.requires_grad, parents=[a, b], op="div")

    def backward_fn(grad: np.ndarray) -> None:
        if a.requires_grad:
            _accumulate_grad(a, _reduce_broadcasted(grad / b.data, a.shape))
        if b.requires_grad:
            _accumulate_grad(b, _reduce_broadcasted(-grad * a.data / (b.data ** 2), b.shape))

    out.backward_fn = backward_fn if out.requires_grad else None
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad, parents=[a, b], op="matmul")

    def backward_fn(grad: np.ndarray) -> None:
        if a.requires_grad:
            _accumulate_grad(a, grad @ b.data.T)
        if b.requires_grad:
            _accumulate_grad(b, a.data.T @ grad)

    out.backward_fn = backward_fn if out.requires_grad else None
    return out


def pow_tensor(a: Tensor, b: Tensor) -> Tensor:
    if b.requires_grad:
        raise NotImplementedError("Gradient for tensor exponent is not implemented in v0 skeleton")
    out = Tensor(a.data ** b.data, requires_grad=a.requires_grad, parents=[a], op="pow")

    def backward_fn(grad: np.ndarray) -> None:
        if a.requires_grad:
            _accumulate_grad(a, grad * b.data * (a.data ** (b.data - 1)))

    out.backward_fn = backward_fn if out.requires_grad else None
    return out


def mean(x: Tensor) -> Tensor:
    out = Tensor(np.array(x.data.mean()), requires_grad=x.requires_grad, parents=[x], op="mean")

    def backward_fn(grad: np.ndarray) -> None:
        if x.requires_grad:
            _accumulate_grad(x, np.ones_like(x.data) * grad / x.data.size)

    out.backward_fn = backward_fn if out.requires_grad else None
    return out


def sum_tensor(x: Tensor) -> Tensor:
    out = Tensor(np.array(x.data.sum()), requires_grad=x.requires_grad, parents=[x], op="sum")

    def backward_fn(grad: np.ndarray) -> None:
        if x.requires_grad:
            _accumulate_grad(x, np.ones_like(x.data) * grad)

    out.backward_fn = backward_fn if out.requires_grad else None
    return out


def detach(x: Tensor) -> Tensor:
    return Tensor(np.array(x.data, copy=True), requires_grad=False)
