# SynapseLang Python Reference Interpreter Skeleton

This repository provides a Python reference interpreter skeleton for **SynapseLang**, a domain-specific programming language for AI, tensors, and differentiable programming.

The goal of this codebase is not to be feature-complete. It is a disciplined starting point for building:

- a lexer
- a parser
- an abstract syntax tree
- a resolver and type checker
- a runtime evaluator
- a NumPy-backed tensor runtime
- a reverse-mode autodiff engine
- a small standard library and CLI

## Status

This is an early skeleton intended to establish structure, interfaces, and implementation direction. Many components are intentionally partial but are wired together in a way that supports steady development.

## Project layout

```text
synapselang-python-ref/
  README.md
  pyproject.toml
  SPEC_NOTES.md
  ROADMAP.md
  examples/
  synapse/
    cli.py
    tokens.py
    lexer.py
    ast.py
    parser.py
    resolver.py
    types.py
    typechecker.py
    runtime/
    autodiff/
    stdlib/
  tests/
```

## Quick start

```bash
python -m synapse.cli tokens examples/autodiff_scalar.syn
python -m synapse.cli ast examples/autodiff_scalar.syn
python -m synapse.cli run examples/autodiff_scalar.syn
```

## Near-term priorities

1. Expand the parser to cover the full v0 grammar.
2. Strengthen type and shape checking.
3. Add function calls to user-defined SynapseLang functions.
4. Expand tensor operations and broadcasting checks.
5. Add more complete `grad` and `value_and_grad` integration.

## Design choices

- **Python** is used for rapid prototyping.
- **NumPy** is used for tensor storage.
- **Reverse-mode autodiff** is implemented in-house to preserve language semantics.
- `grad` and `value_and_grad` are treated as prelude built-ins.

## Example

```synapse
fn f(x: Float32) -> Float32 {
    return x * x + 3.0 * x;
}

let y = f(2.0);
print(y);
```

## License

This project is licensed under the Apache License 2.0.

You may use, modify, and distribute this software under the terms of the Apache 2.0 license.

See the [LICENSE](./LICENSE) file for full details.
