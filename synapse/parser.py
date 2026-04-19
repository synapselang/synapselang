from __future__ import annotations

from dataclasses import dataclass

from . import ast
from .tokens import Token, TokenKind


class ParseError(Exception):
    """Raised when the parser encounters invalid syntax."""


@dataclass
class Parser:
    tokens: list[Token]
    current: int = 0

    def parse(self) -> ast.Program:
        items: list[ast.Node] = []
        while not self._is_at_end():
            if self._match(TokenKind.FN):
                items.append(self._function_decl())
            elif self._match(TokenKind.STRUCT):
                items.append(self._struct_decl())
            else:
                items.append(self._statement())
        return ast.Program(items)

    def _function_decl(self) -> ast.FunctionDecl:
        name = self._consume(TokenKind.IDENT, "Expected function name").lexeme
        self._consume(TokenKind.LPAREN, "Expected '('")
        params: list[str] = []
        if not self._check(TokenKind.RPAREN):
            while True:
                param_name = self._consume(TokenKind.IDENT, "Expected parameter name").lexeme
                if self._match(TokenKind.COLON):
                    self._type_name()
                params.append(param_name)
                if not self._match(TokenKind.COMMA):
                    break
        self._consume(TokenKind.RPAREN, "Expected ')'")
        return_type = None
        if self._match(TokenKind.ARROW):
            return_type = self._type_name()
        body = self._block()
        return ast.FunctionDecl(name=name, params=params, body=body, return_type=return_type)

    def _struct_decl(self) -> ast.StructDecl:
        name = self._consume(TokenKind.IDENT, "Expected struct name").lexeme
        self._consume(TokenKind.LBRACE, "Expected '{'")
        fields: list[ast.StructField] = []
        while not self._check(TokenKind.RBRACE):
            is_param = self._match(TokenKind.PARAM)
            field_name = self._consume(TokenKind.IDENT, "Expected field name").lexeme
            type_name = None
            if self._match(TokenKind.COLON):
                type_name = self._type_name()
            fields.append(ast.StructField(name=field_name, type_name=type_name, is_param=is_param))
            self._match(TokenKind.COMMA)
        self._consume(TokenKind.RBRACE, "Expected '}'")
        return ast.StructDecl(name=name, fields=fields)

    def _statement(self) -> ast.Statement:
        if self._match(TokenKind.LET):
            name = self._consume(TokenKind.IDENT, "Expected variable name").lexeme
            if self._match(TokenKind.COLON):
                self._type_name()
            self._consume(TokenKind.EQ, "Expected '='")
            value = self._expression()
            self._consume(TokenKind.SEMI, "Expected ';'")
            return ast.LetStmt(name=name, value=value)
        if self._match(TokenKind.RETURN):
            value = None if self._check(TokenKind.SEMI) else self._expression()
            self._consume(TokenKind.SEMI, "Expected ';'")
            return ast.ReturnStmt(value=value)
        if self._match(TokenKind.LBRACE):
            return self._block_from_open_brace()
        expr = self._expression()
        self._consume(TokenKind.SEMI, "Expected ';'")
        return ast.ExprStmt(expr)

    def _block(self) -> ast.Block:
        self._consume(TokenKind.LBRACE, "Expected '{'")
        return self._block_from_open_brace()

    def _block_from_open_brace(self) -> ast.Block:
        statements: list[ast.Statement] = []
        while not self._check(TokenKind.RBRACE) and not self._is_at_end():
            statements.append(self._statement())
        self._consume(TokenKind.RBRACE, "Expected '}'")
        return ast.Block(statements=statements)

    def _expression(self) -> ast.Expression:
        return self._addition()

    def _addition(self) -> ast.Expression:
        expr = self._multiplication()
        while self._match(TokenKind.PLUS, TokenKind.MINUS):
            op = self._previous().lexeme
            right = self._multiplication()
            expr = ast.Binary(expr, op, right)
        return expr

    def _multiplication(self) -> ast.Expression:
        expr = self._call()
        while self._match(TokenKind.STAR, TokenKind.SLASH, TokenKind.AT):
            op = self._previous().lexeme
            right = self._call()
            expr = ast.Binary(expr, op, right)
        return expr

    def _call(self) -> ast.Expression:
        expr = self._primary()
        while True:
            if self._match(TokenKind.LPAREN):
                args: list[ast.Expression] = []
                if not self._check(TokenKind.RPAREN):
                    while True:
                        args.append(self._expression())
                        if not self._match(TokenKind.COMMA):
                            break
                self._consume(TokenKind.RPAREN, "Expected ')'")
                expr = ast.Call(expr, args)
            elif self._match(TokenKind.DOT):
                name = self._consume(TokenKind.IDENT, "Expected member name").lexeme
                expr = ast.Member(expr, name)
            else:
                break
        return expr

    def _primary(self) -> ast.Expression:
        if self._match(TokenKind.INT):
            return ast.Literal(int(self._previous().lexeme))
        if self._match(TokenKind.FLOAT):
            return ast.Literal(float(self._previous().lexeme))
        if self._match(TokenKind.STRING):
            return ast.Literal(self._previous().lexeme)
        if self._match(TokenKind.TRUE):
            return ast.Literal(True)
        if self._match(TokenKind.FALSE):
            return ast.Literal(False)
        if self._match(TokenKind.IDENT):
            return ast.Identifier(self._previous().lexeme)
        if self._match(TokenKind.LPAREN):
            expr = self._expression()
            self._consume(TokenKind.RPAREN, "Expected ')'")
            return expr
        raise ParseError(f"Unexpected token {self._peek()}")

    def _type_name(self) -> str:
        name = self._consume(TokenKind.IDENT, "Expected type name").lexeme
        if self._match(TokenKind.LT):
            depth = 1
            pieces = [name, "<"]
            while depth > 0:
                tok = self._advance()
                pieces.append(tok.lexeme)
                if tok.kind == TokenKind.LT:
                    depth += 1
                elif tok.kind == TokenKind.GT:
                    depth -= 1
            name = "".join(pieces)
        if self._match(TokenKind.LBRACKET):
            pieces = [name, "["]
            while not self._check(TokenKind.RBRACKET):
                pieces.append(self._advance().lexeme)
            self._consume(TokenKind.RBRACKET, "Expected ']'")
            pieces.append("]")
            name = "".join(pieces)
        return name

    def _match(self, *kinds: TokenKind) -> bool:
        for kind in kinds:
            if self._check(kind):
                self._advance()
                return True
        return False

    def _consume(self, kind: TokenKind, message: str) -> Token:
        if self._check(kind):
            return self._advance()
        raise ParseError(f"{message}: got {self._peek()}")

    def _check(self, kind: TokenKind) -> bool:
        if self._is_at_end():
            return kind == TokenKind.EOF
        return self._peek().kind == kind

    def _advance(self) -> Token:
        if not self._is_at_end():
            self.current += 1
        return self._previous()

    def _is_at_end(self) -> bool:
        return self._peek().kind == TokenKind.EOF

    def _peek(self) -> Token:
        return self.tokens[self.current]

    def _previous(self) -> Token:
        return self.tokens[self.current - 1]


def parse(tokens: list[Token]) -> ast.Program:
    return Parser(tokens).parse()
