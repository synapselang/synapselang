from __future__ import annotations

from dataclasses import dataclass, field

from . import ast


class ResolveError(Exception):
    """Raised for invalid name resolution."""


@dataclass
class Scope:
    names: set[str] = field(default_factory=set)


class Resolver:
    def __init__(self) -> None:
        self.scopes: list[Scope] = []
        self.globals: set[str] = set()

    def resolve(self, program: ast.Program) -> None:
        self.begin_scope()
        for builtin in {"print", "tensor", "grad", "value_and_grad", "mean", "sum"}:
            self.declare(builtin)
        for item in program.items:
            if isinstance(item, ast.FunctionDecl | ast.StructDecl):
                self.declare(item.name)
        for item in program.items:
            self._resolve_node(item)
        self.end_scope()

    def _resolve_node(self, node: ast.Node) -> None:
        if isinstance(node, ast.FunctionDecl):
            self.begin_scope()
            for param in node.params:
                self.declare(param)
            self._resolve_block(node.body)
            self.end_scope()
        elif isinstance(node, ast.StructDecl):
            return
        elif isinstance(node, ast.LetStmt):
            self._resolve_expr(node.value)
            self.declare(node.name)
        elif isinstance(node, ast.ExprStmt):
            self._resolve_expr(node.expression)
        elif isinstance(node, ast.ReturnStmt) and node.value is not None:
            self._resolve_expr(node.value)
        elif isinstance(node, ast.Block):
            self.begin_scope()
            self._resolve_block(node)
            self.end_scope()

    def _resolve_block(self, block: ast.Block) -> None:
        for stmt in block.statements:
            self._resolve_node(stmt)

    def _resolve_expr(self, expr: ast.Expression) -> None:
        if isinstance(expr, ast.Identifier):
            if not self.is_declared(expr.name):
                raise ResolveError(f"Undefined name: {expr.name}")
        elif isinstance(expr, ast.Binary):
            self._resolve_expr(expr.left)
            self._resolve_expr(expr.right)
        elif isinstance(expr, ast.Unary):
            self._resolve_expr(expr.operand)
        elif isinstance(expr, ast.Call):
            self._resolve_expr(expr.callee)
            for arg in expr.arguments:
                self._resolve_expr(arg)
        elif isinstance(expr, ast.Member):
            self._resolve_expr(expr.object)
        elif isinstance(expr, ast.Literal):
            return

    def begin_scope(self) -> None:
        self.scopes.append(Scope())

    def end_scope(self) -> None:
        self.scopes.pop()

    def declare(self, name: str) -> None:
        if not self.scopes:
            self.globals.add(name)
            return
        scope = self.scopes[-1]
        if name in scope.names:
            raise ResolveError(f"Duplicate declaration: {name}")
        scope.names.add(name)

    def is_declared(self, name: str) -> bool:
        return any(name in scope.names for scope in reversed(self.scopes)) or name in self.globals
