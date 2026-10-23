"""
boolean_model.py: the Boolean retrieval model.

- Incidence matrix  M[t][d] = 1 if term t occurs in document d, else 0.
  Boolean operators become bitwise operations on the rows of the matrix.
- Inverted index    term -> posting list. The positional form records, for each document,
  the term frequency and the positions:  <id_j, f_ij, [O1, O2, ..., Ok]>.
- Query parser      AND, OR, NOT and parentheses with the precedence
                    ()  >  NOT  >  AND  >  OR.

This module does not depend on Streamlit. Try it from the terminal:
    python boolean_model.py "Brutus AND Caesar AND NOT Calpurnia"
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OPERATORS = {"NOT": 3, "AND": 2, "OR": 1}          # higher binds tighter


def load_collections() -> dict:
    return json.loads((DATA_DIR / "collections.json").read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    """Lowercase words (letters and digits). No stopword removal, so positions match the text."""
    return re.findall(r"[^\W_]+", text.lower())


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

@dataclass
class Posting:
    doc: str
    freq: int
    positions: list[int]                          # 1-based, as in the lecture

    def __str__(self) -> str:
        return f"<{self.doc}, {self.freq}, [{', '.join(map(str, self.positions))}]>"


class BooleanIndex:
    def __init__(self, documents: dict[str, str]):
        self.docs = list(documents)               # document order = bit order
        self.texts = documents
        self.index: dict[str, list[Posting]] = {}
        for doc, text in documents.items():
            positions: dict[str, list[int]] = {}
            for pos, term in enumerate(tokenize(text), start=1):
                positions.setdefault(term, []).append(pos)
            for term, pos_list in positions.items():
                self.index.setdefault(term, []).append(Posting(doc, len(pos_list), pos_list))
        self.terms = sorted(self.index)

    # --- inverted index -----------------------------------------------------
    def postings(self, term: str) -> list[str]:
        """Document IDs that contain the term, in document order (a sorted posting list)."""
        return [p.doc for p in self.index.get(term.lower(), [])]

    def df(self, term: str) -> int:
        return len(self.index.get(term.lower(), []))

    # --- incidence matrix ---------------------------------------------------
    def bits(self, term: str) -> list[int]:
        """Row of the incidence matrix for a term."""
        present = set(self.postings(term))
        return [1 if d in present else 0 for d in self.docs]

    def matrix(self) -> list[list[int]]:
        return [self.bits(t) for t in self.terms]

    def size_stats(self) -> dict[str, int]:
        cells = len(self.terms) * len(self.docs)
        postings = sum(len(p) for p in self.index.values())
        return {"terms": len(self.terms), "documents": len(self.docs), "matrix cells": cells,
                "ones (postings)": postings, "zeros": cells - postings}

    # --- phrase query with positions -----------------------------------------
    def phrase(self, phrase: str) -> list[tuple[str, list[int]]]:
        """Documents where the words occur next to each other, with the start positions."""
        words = tokenize(phrase)
        if not words or any(w not in self.index for w in words):
            return []
        pos = {w: {p.doc: set(p.positions) for p in self.index[w]} for w in words}
        hits = []
        for doc in self.postings(words[0]):
            starts = [s for s in pos[words[0]][doc]
                      if all(s + k in pos[w].get(doc, ()) for k, w in enumerate(words[1:], 1))]
            if starts:
                hits.append((doc, sorted(starts)))
        return hits


# ---------------------------------------------------------------------------
# Query parsing (shunting-yard) and evaluation
# ---------------------------------------------------------------------------

class QueryError(ValueError):
    pass


def lex(query: str) -> list[str]:
    tokens = re.findall(r"\(|\)|[^\s()]+", query)
    out = []
    for t in tokens:
        up = t.upper()
        out.append(up if up in OPERATORS else t.lower() if t not in "()" else t)
    # implicit AND between two operands, e.g. "brutus caesar" or "brutus (caesar OR x)"
    result: list[str] = []
    for t in out:
        if result and _ends_operand(result[-1]) and (_is_term(t) or t in ("(", "NOT")):
            result.append("AND")
        result.append(t)
    return result


def _is_term(t: str) -> bool:
    return t not in OPERATORS and t not in ("(", ")")


def _ends_operand(t: str) -> bool:
    return _is_term(t) or t == ")"


def to_postfix(tokens: list[str]) -> list[str]:
    """Shunting-yard: infix tokens -> postfix (reverse Polish) order, honouring precedence."""
    output, stack = [], []
    expect_operand = True
    for t in tokens:
        if _is_term(t):
            if not expect_operand:
                raise QueryError(f"missing operator before '{t}'")
            output.append(t)
            expect_operand = False
        elif t == "NOT":
            if not expect_operand:
                raise QueryError("NOT must come before a term or a bracket")
            stack.append(t)
        elif t in ("AND", "OR"):
            if expect_operand:
                raise QueryError(f"{t} needs a term on its left")
            while stack and stack[-1] in OPERATORS and OPERATORS[stack[-1]] >= OPERATORS[t]:
                output.append(stack.pop())
            stack.append(t)
            expect_operand = True
        elif t == "(":
            if not expect_operand:
                raise QueryError("missing operator before '('")
            stack.append(t)
        elif t == ")":
            if expect_operand:
                raise QueryError("empty brackets or an operator right before ')'")
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise QueryError("there is a ')' without a matching '('")
            stack.pop()
    if expect_operand:
        raise QueryError("the query ends with an operator" if tokens else "the query is empty")
    while stack:
        op = stack.pop()
        if op == "(":
            raise QueryError("there is a '(' without a matching ')'")
        output.append(op)
    return output


@dataclass
class Step:
    expression: str        # e.g. "brutus AND caesar"
    operation: str         # e.g. "110100 AND 110111"
    bits: list[int]        # result
    sets: str              # e.g. "posting(brutus) ∩ posting(caesar)"


@dataclass
class QueryResult:
    tokens: list[str]
    postfix: list[str]
    bracketed: str
    steps: list[Step]
    bits: list[int]
    docs: list[str]
    unknown_terms: list[str]


def _bitstr(bits: list[int]) -> str:
    return "".join(map(str, bits))


def evaluate(query: str, idx: BooleanIndex) -> QueryResult:
    tokens = lex(query)
    postfix = to_postfix(tokens)
    stack: list[tuple[str, list[int], str]] = []     # (expression, bits, set notation)
    steps: list[Step] = []
    for t in postfix:
        if _is_term(t):
            stack.append((t, idx.bits(t), f"posting({t})"))
        elif t == "NOT":
            e, b, s = stack.pop()
            res = [1 - x for x in b]
            expr = f"NOT {e}"
            steps.append(Step(expr, f"NOT {_bitstr(b)}", res, f"U \\ {s}"))
            stack.append((f"(NOT {e})" if " " in e else expr, res, f"(U \\ {s})"))
        else:
            (e2, b2, s2), (e1, b1, s1) = stack.pop(), stack.pop()
            res = [x & y for x, y in zip(b1, b2)] if t == "AND" else [x | y for x, y in zip(b1, b2)]
            expr = f"{e1} {t} {e2}"
            sym = "∩" if t == "AND" else "∪"
            steps.append(Step(expr, f"{_bitstr(b1)} {t} {_bitstr(b2)}", res, f"{s1} {sym} {s2}"))
            stack.append((f"({expr})", res, f"({s1} {sym} {s2})"))
    expr, bits, _ = stack[-1]
    bracketed = expr[1:-1] if expr.startswith("(") and expr.endswith(")") and len(steps) else expr
    unknown = sorted({t for t in postfix if _is_term(t) and t not in idx.index})
    return QueryResult(tokens, postfix, bracketed, steps, bits,
                       [d for d, b in zip(idx.docs, bits) if b], unknown)


# ---------------------------------------------------------------------------
# Merge intersection (the lecture pseudocode), with a trace
# ---------------------------------------------------------------------------

def intersect_trace(p1: list[str], p2: list[str], order: list[str]) -> tuple[list[str], list[dict]]:
    """
    FUNCTION BooleanAND(p1, p2): walk both sorted posting lists with two pointers.
    Returns the intersection and one row per comparison. Cost: O(|p1| + |p2|).
    """
    rank = {d: i for i, d in enumerate(order)}
    result, rows, i, j = [], [], 0, 0
    while i < len(p1) and j < len(p2):
        a, b = p1[i], p2[j]
        if a == b:
            action = f"equal → add {a}, i++, j++"
            result.append(a)
            i, j = i + 1, j + 1
        elif rank[a] < rank[b]:
            action = f"{a} < {b} → i++"
            i += 1
        else:
            action = f"{a} > {b} → j++"
            j += 1
        rows.append({"Step": len(rows) + 1, "p1[i]": a, "p2[j]": b, "Action": action,
                     "Result so far": ", ".join(result) or "–"})
    return result, rows


def and_plan(terms: list[str], idx: BooleanIndex) -> list[dict]:
    """Optimization for t1 AND t2 AND ...: start with the shortest posting list."""
    ordered = sorted(dict.fromkeys(t.lower() for t in terms), key=idx.df)
    rows, current = [], None
    for t in ordered:
        p = idx.postings(t)
        current = p if current is None else [d for d in current if d in set(p)]
        rows.append({"Term": t, "df": len(p), "Posting list": ", ".join(p) or "–",
                     "Candidates left": ", ".join(current) or "– (stop: the result is empty)"})
        if not current:
            break
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run a Boolean query against a lecture collection.")
    ap.add_argument("query", nargs="?", default="Brutus AND Caesar AND NOT Calpurnia")
    ap.add_argument("--collection", type=int, default=0, help="0-3, see data/collections.json")
    args = ap.parse_args()

    name, col = list(load_collections().items())[args.collection]
    idx = BooleanIndex(col["documents"])
    r = evaluate(args.query, idx)
    print(f"Collection : {name}")
    print(f"Documents  : {' | '.join(idx.docs)}")
    print(f"Query      : {args.query}")
    print(f"Precedence : {r.bracketed}")
    print(f"Postfix    : {' '.join(r.postfix)}")
    for s in r.steps:
        print(f"  {s.expression:<40} {s.operation} = {_bitstr(s.bits)}")
    print(f"Result     : {_bitstr(r.bits)} -> {', '.join(r.docs) or '(no documents)'}")
