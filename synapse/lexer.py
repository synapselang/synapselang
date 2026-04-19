from __future__ import annotations

from dataclasses import dataclass

from .tokens import KEYWORDS, Span, Token, TokenKind


class LexError(Exception):
    """Raised when the lexer encounters invalid source."""


@dataclass
class Lexer:
    source: str
    index: int = 0
    line: int = 1
    column: int = 1

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while not self._is_at_end():
            self._skip_whitespace_and_comments()
            if self._is_at_end():
                break
            tokens.append(self._scan_token())
        tokens.append(Token(TokenKind.EOF, "", Span(self.line, self.column)))
        return tokens

    def _scan_token(self) -> Token:
        start_line, start_col = self.line, self.column
        c = self._advance()

        single = {
            "(": TokenKind.LPAREN,
            ")": TokenKind.RPAREN,
            "{": TokenKind.LBRACE,
            "}": TokenKind.RBRACE,
            "[": TokenKind.LBRACKET,
            "]": TokenKind.RBRACKET,
            ",": TokenKind.COMMA,
            ":": TokenKind.COLON,
            ";": TokenKind.SEMI,
            "+": TokenKind.PLUS,
            "*": TokenKind.STAR,
            "/": TokenKind.SLASH,
            "%": TokenKind.PERCENT,
            "@": TokenKind.AT,
        }
        if c in single:
            if c == "*" and self._match("*"):
                return Token(TokenKind.POW, "**", Span(start_line, start_col))
            return Token(single[c], c, Span(start_line, start_col))

        if c == ".":
            if self._match("."):
                return Token(TokenKind.RANGE, "..", Span(start_line, start_col))
            return Token(TokenKind.DOT, c, Span(start_line, start_col))
        if c == "-":
            if self._match(">"):
                return Token(TokenKind.ARROW, "->", Span(start_line, start_col))
            return Token(TokenKind.MINUS, c, Span(start_line, start_col))
        if c == "=":
            if self._match("="):
                return Token(TokenKind.EQEQ, "==", Span(start_line, start_col))
            return Token(TokenKind.EQ, c, Span(start_line, start_col))
        if c == "!":
            if self._match("="):
                return Token(TokenKind.NEQ, "!=", Span(start_line, start_col))
            return Token(TokenKind.BANG, c, Span(start_line, start_col))
        if c == "<":
            if self._match("="):
                return Token(TokenKind.LTE, "<=", Span(start_line, start_col))
            return Token(TokenKind.LT, c, Span(start_line, start_col))
        if c == ">":
            if self._match("="):
                return Token(TokenKind.GTE, ">=", Span(start_line, start_col))
            return Token(TokenKind.GT, c, Span(start_line, start_col))
        if c == "&" and self._match("&"):
            return Token(TokenKind.ANDAND, "&&", Span(start_line, start_col))
        if c == "|" and self._match("|"):
            return Token(TokenKind.OROR, "||", Span(start_line, start_col))
        if c == '"':
            return self._string(start_line, start_col)
        if c.isdigit():
            return self._number(c, start_line, start_col)
        if c.isalpha() or c == "_":
            return self._identifier(c, start_line, start_col)

        raise LexError(f"Unexpected character {c!r} at {start_line}:{start_col}")

    def _string(self, start_line: int, start_col: int) -> Token:
        chars: list[str] = []
        while not self._is_at_end() and self._peek() != '"':
            chars.append(self._advance())
        if self._is_at_end():
            raise LexError(f"Unterminated string at {start_line}:{start_col}")
        self._advance()
        value = "".join(chars)
        return Token(TokenKind.STRING, value, Span(start_line, start_col))

    def _number(self, first: str, start_line: int, start_col: int) -> Token:
        chars = [first]
        is_float = False
        while not self._is_at_end() and self._peek().isdigit():
            chars.append(self._advance())
        if not self._is_at_end() and self._peek() == "." and self._peek_next().isdigit():
            is_float = True
            chars.append(self._advance())
            while not self._is_at_end() and self._peek().isdigit():
                chars.append(self._advance())
        kind = TokenKind.FLOAT if is_float else TokenKind.INT
        return Token(kind, "".join(chars), Span(start_line, start_col))

    def _identifier(self, first: str, start_line: int, start_col: int) -> Token:
        chars = [first]
        while not self._is_at_end() and (self._peek().isalnum() or self._peek() == "_"):
            chars.append(self._advance())
        lexeme = "".join(chars)
        kind = KEYWORDS.get(lexeme, TokenKind.IDENT)
        return Token(kind, lexeme, Span(start_line, start_col))

    def _skip_whitespace_and_comments(self) -> None:
        while not self._is_at_end():
            c = self._peek()
            if c in " \t\r":
                self._advance()
                continue
            if c == "\n":
                self._advance()
                continue
            if c == "/" and self._peek_next() == "/":
                while not self._is_at_end() and self._peek() != "\n":
                    self._advance()
                continue
            if c == "/" and self._peek_next() == "*":
                self._advance()
                self._advance()
                while not self._is_at_end() and not (self._peek() == "*" and self._peek_next() == "/"):
                    self._advance()
                if self._is_at_end():
                    raise LexError("Unterminated block comment")
                self._advance()
                self._advance()
                continue
            return

    def _peek(self) -> str:
        return self.source[self.index]

    def _peek_next(self) -> str:
        if self.index + 1 >= len(self.source):
            return "\0"
        return self.source[self.index + 1]

    def _advance(self) -> str:
        c = self.source[self.index]
        self.index += 1
        if c == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return c

    def _match(self, expected: str) -> bool:
        if self._is_at_end() or self.source[self.index] != expected:
            return False
        self._advance()
        return True

    def _is_at_end(self) -> bool:
        return self.index >= len(self.source)


def lex(source: str) -> list[Token]:
    return Lexer(source).tokenize()
