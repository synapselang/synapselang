from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Node:
    pass


class Statement(Node):
    pass


class Expression(Node):
    pass


@dataclass(slots=True)
class Program(Node):
    items: list[Node]


@dataclass(slots=True)
class Identifier(Expression):
    name: str


@dataclass(slots=True)
class Literal(Expression):
    value: Any


@dataclass(slots=True)
class Binary(Expression):
    left: Expression
    operator: str
    right: Expression


@dataclass(slots=True)
class Unary(Expression):
    operator: str
    operand: Expression


@dataclass(slots=True)
class Call(Expression):
    callee: Expression
    arguments: list[Expression]


@dataclass(slots=True)
class Member(Expression):
    object: Expression
    name: str


@dataclass(slots=True)
class LetStmt(Statement):
    name: str
    value: Expression


@dataclass(slots=True)
class ExprStmt(Statement):
    expression: Expression


@dataclass(slots=True)
class ReturnStmt(Statement):
    value: Expression | None


@dataclass(slots=True)
class Block(Statement):
    statements: list[Statement] = field(default_factory=list)


@dataclass(slots=True)
class FunctionDecl(Node):
    name: str
    params: list[str]
    body: Block
    return_type: str | None = None


@dataclass(slots=True)
class StructField(Node):
    name: str
    type_name: str | None = None
    is_param: bool = False


@dataclass(slots=True)
class StructDecl(Node):
    name: str
    fields: list[StructField]
