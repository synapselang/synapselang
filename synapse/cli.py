from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

from synapse.lexer import lex
from synapse.parser import parse
from synapse.resolver import Resolver
from synapse.runtime.evaluator import Evaluator
from synapse.typechecker import TypeChecker


def load_source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def cmd_tokens(path: str) -> int:
    tokens = lex(load_source(path))
    for token in tokens:
        print(token)
    return 0


def cmd_ast(path: str) -> int:
    program = parse(lex(load_source(path)))
    pprint(program)
    return 0


def cmd_check(path: str) -> int:
    program = parse(lex(load_source(path)))
    Resolver().resolve(program)
    TypeChecker().check(program)
    print("check passed")
    return 0


def cmd_run(path: str) -> int:
    source = load_source(path)
    program = parse(lex(source))
    Resolver().resolve(program)
    TypeChecker().check(program)
    Evaluator.with_prelude().evaluate_program(program)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synapse")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("tokens", "ast", "check", "run"):
        p = sub.add_parser(name)
        p.add_argument("path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "tokens":
        return cmd_tokens(args.path)
    if args.command == "ast":
        return cmd_ast(args.path)
    if args.command == "check":
        return cmd_check(args.path)
    if args.command == "run":
        return cmd_run(args.path)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
