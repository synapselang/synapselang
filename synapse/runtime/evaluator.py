from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from synapse import ast
from synapse.runtime.builtins import make_builtins
from synapse.runtime.environment import Environment
from synapse.runtime.values import BuiltinFunction, StructType, UserFunction


class ReturnSignal(Exception):
    def __init__(self, value: object) -> None:
        self.value = value
        super().__init__("return")


@dataclass
class Evaluator:
    globals: Environment

    @classmethod
    def with_prelude(cls) -> "Evaluator":
        env = Environment()
        for name, value in make_builtins().items():
            env.define(name, value)
        return cls(globals=env)

    def evaluate_program(self, program: ast.Program) -> Any:
        env = Environment(parent=self.globals)
        for item in program.items:
            if isinstance(item, ast.StructDecl):
                env.define(item.name, StructType(item.name, [field.name for field in item.fields]))
            elif isinstance(item, ast.FunctionDecl):
                env.define(item.name, UserFunction(item.name, item.params, item.body, env))
            else:
                self._exec_stmt(item, env)
        return None

    def _exec_stmt(self, stmt: ast.Node, env: Environment) -> Any:
        if isinstance(stmt, ast.LetStmt):
            env.define(stmt.name, self._eval_expr(stmt.value, env))
            return None
        if isinstance(stmt, ast.ExprStmt):
            return self._eval_expr(stmt.expression, env)
        if isinstance(stmt, ast.ReturnStmt):
            value = None if stmt.value is None else self._eval_expr(stmt.value, env)
            raise ReturnSignal(value)
        if isinstance(stmt, ast.Block):
            block_env = Environment(parent=env)
            result = None
            for inner in stmt.statements:
                result = self._exec_stmt(inner, block_env)
            return result
        raise NotImplementedError(f"Unsupported statement: {type(stmt).__name__}")

    def _eval_expr(self, expr: ast.Expression, env: Environment) -> Any:
        if isinstance(expr, ast.Literal):
            return expr.value
        if isinstance(expr, ast.Identifier):
            return env.get(expr.name)
        if isinstance(expr, ast.Binary):
            left = self._eval_expr(expr.left, env)
            right = self._eval_expr(expr.right, env)
            return self._apply_binary(left, expr.operator, right)
        if isinstance(expr, ast.Call):
            callee = self._eval_expr(expr.callee, env)
            args = [self._eval_expr(arg, env) for arg in expr.arguments]
            return self._call(callee, args)
        if isinstance(expr, ast.Member):
            obj = self._eval_expr(expr.object, env)
            return getattr(obj, expr.name)
        raise NotImplementedError(f"Unsupported expression: {type(expr).__name__}")

    def _call(self, callee: Any, args: list[Any]) -> Any:
        if isinstance(callee, BuiltinFunction):
            return callee(*args)
        if isinstance(callee, UserFunction):
            call_env = Environment(parent=callee.closure)
            for name, value in zip(callee.params, args, strict=False):
                call_env.define(name, value)
            try:
                self._exec_stmt(callee.body, call_env)
            except ReturnSignal as signal:
                return signal.value
            return None
        raise TypeError(f"Value is not callable: {callee!r}")

    def _apply_binary(self, left: Any, operator: str, right: Any) -> Any:
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/":
            return left / right
        if operator == "@":
            return left @ right
        raise TypeError(f"Unsupported operator: {operator}")
