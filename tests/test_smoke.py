from synapse.lexer import lex
from synapse.parser import parse


def test_smoke_parse() -> None:
    source = "let x = 1 + 2;"
    program = parse(lex(source))
    assert len(program.items) == 1
