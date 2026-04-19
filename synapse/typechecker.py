from __future__ import annotations

from . import ast
from .types import BOOL, FLOAT32, FLOAT64, INT64, UNKNOWN, ScalarType, Type


class TypeErrorSynapse(Exception):
    """Raised when a static type check fails."""


class TypeChecker:
    def check(self, program: ast.Program) -> None:
        env: dict[str, Type] = {
            "print": UNKNOWN,
            "tensor": UNKNOWN,
            "mean": UNKNOWN,
            "sum": UNKNOWN,
            "grad": UNKNOWN,
            "value_and_grad": UNKNOWN,
        }
        for item in program.items:
            if isinstance(item, ast.FunctionDecl):
                env[item.name] = UNKNOWN
            elif isinstance(item, ast.StructDecl):
                env[item.name] = UNKNOWN
        for item in program.items:
            self._check_node(item, env)

    def _check_node(self, node: ast.Node, env: dict[str, Type]) -> Type | None:
        if isinstance(node, ast.LetStmt):
            t = self._infer_expr(node.value, env)
            env[node.name] = t
            return None
        if isinstance(node, ast.ExprStmt):
            return self._infer_expr(node.expression, env)
        if isinstance(node, ast.ReturnStmt):
            if node.value is not None:
                return self._infer_expr(node.value, env)
            return None
        if isinstance(node, ast.FunctionDecl):
            local = env.copy()
            for param in node.params:
                local[param] = UNKNOWN
            for stmt in node.body.statements:
                self._check_node(stmt, local)
        return None

    def _infer_expr(self, expr: ast.Expression, env: dict[str, Type]) -> Type:
        if isinstance(expr, ast.Literal):
            if isinstance(expr.value, bool):
                return BOOL
            if isinstance(expr.value, int):
                return INT64
            if isinstance(expr.value, float):
                return FLOAT64
            return UNKNOWN
        if isinstance(expr, ast.Identifier):
            return env.get(expr.name, UNKNOWN)
        if isinstance(expr, ast.Binary):
            left = self._infer_expr(expr.left, env)
            right = self._infer_expr(expr.right, env)
            if expr.operator in {"+", "-", "*", "/"}:
                return self._promote(left, right)
            if expr.operator == "@":
                return UNKNOWN
            raise TypeErrorSynapse(f"Unsupported operator: {expr.operator}")
        if isinstance(expr, ast.Call):
            return UNKNOWN
        if isinstance(expr, ast.Member):
            return UNKNOWN
        raise TypeErrorSynapse(f"Unknown expression node: {type(expr).__name__}")

    def _promote(self, left: Type, right: Type) -> Type:
        if left == FLOAT64 or right == FLOAT64:
            return FLOAT64
        if left == FLOAT32 or right == FLOAT32:
            return FLOAT32
        if isinstance(left, ScalarType) and isinstance(right, ScalarType):
            return left
        return UNKNOWN
