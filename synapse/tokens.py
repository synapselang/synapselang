from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenKind(Enum):
    EOF = auto()
    IDENT = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()

    LET = auto()
    VAR = auto()
    FN = auto()
    STRUCT = auto()
    PARAM = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    IN = auto()
    IMPORT = auto()
    TRUE = auto()
    FALSE = auto()

    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    SEMI = auto()
    ARROW = auto()
    RANGE = auto()

    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    AT = auto()
    POW = auto()
    BANG = auto()

    EQ = auto()
    EQEQ = auto()
    NEQ = auto()
    LT = auto()
    LTE = auto()
    GT = auto()
    GTE = auto()
    ANDAND = auto()
    OROR = auto()


KEYWORDS: dict[str, TokenKind] = {
    "let": TokenKind.LET,
    "var": TokenKind.VAR,
    "fn": TokenKind.FN,
    "struct": TokenKind.STRUCT,
    "param": TokenKind.PARAM,
    "return": TokenKind.RETURN,
    "if": TokenKind.IF,
    "else": TokenKind.ELSE,
    "for": TokenKind.FOR,
    "in": TokenKind.IN,
    "import": TokenKind.IMPORT,
    "true": TokenKind.TRUE,
    "false": TokenKind.FALSE,
}


@dataclass(slots=True, frozen=True)
class Span:
    line: int
    column: int


@dataclass(slots=True, frozen=True)
class Token:
    kind: TokenKind
    lexeme: str
    span: Span

    def __str__(self) -> str:
        return f"{self.kind.name}({self.lexeme!r} at {self.span.line}:{self.span.column})"
