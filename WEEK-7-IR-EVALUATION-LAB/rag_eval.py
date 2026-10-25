"""
rag_eval.py: simplified, LLM-free stand-ins for the four RAGAS metrics (Es et al., EACL 2024).

RAGAS scores a retrieval-augmented generation (RAG) answer from the question, the retrieved contexts and the
answer itself. The real library asks an LLM to split the answer into claims and to judge each one. Without an
LLM, this module approximates the same quantities so their meaning can be explored:

  Faithfulness       share of answer sentences supported by the contexts.
                     A sentence counts as supported when at least `tau` of its content words occur in the
                     contexts (RAGAS: an LLM checks whether each claim can be inferred from the context).
  Answer relevance   cosine similarity between the question and the answer embeddings, clipped to [0, 1]
                     (RAGAS: the LLM writes questions the answer would fit, compared with the real question).
  Context precision  Σ_k P@k · v_k / (relevant contexts in the top K), where v_k = 1 if context k is relevant
                     (here relevance comes from the qrels; RAGAS asks the LLM).
  Context recall     share of the reference sentences supported by the contexts. The reference answer is
                     the text of the documents judged relevant (grade ≥ 2).

These are teaching approximations: word overlap cannot detect a negation or a paraphrase the way an LLM judge can.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from systems import tokenize

FABRICATED = {
    "Search technology": "Google invented this technique in 2015, and since then no search engine has used any other method.",
    "Health & nutrition": "A 2025 study of ten million people found that a single daily dose removes the risk completely.",
    "Personal finance": "Experts guarantee a yearly return of 25 percent with no risk at all.",
}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", text.strip()) if s.strip()]


def _norm(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def content_words(text: str) -> set[str]:
    return {_norm(w) for w in tokenize(text)}


@dataclass
class Support:
    sentence: str
    words: list[str]
    found: list[str]

    @property
    def ratio(self) -> float:
        return len(self.found) / len(self.words) if self.words else 0.0


def support(sentence: str, contexts: list[str]) -> Support:
    pool = set().union(*(content_words(c) for c in contexts)) if contexts else set()
    words = sorted(content_words(sentence))
    return Support(sentence, words, [w for w in words if w in pool])


def faithfulness(answer: str, contexts: list[str], tau: float = 0.6) -> tuple[float, list[Support]]:
    sup = [support(s, contexts) for s in sentences(answer)]
    return (sum(s.ratio >= tau for s in sup) / len(sup) if sup else 0.0), sup


def answer_relevance(question: str, answer: str, encode) -> float:
    if not answer.strip():
        return 0.0
    v = encode([question, answer])
    return float(max(0.0, min(1.0, v[0] @ v[1])))


def context_precision(relevant_flags: list[bool]) -> float:
    hits, total = 0, 0.0
    for k, rel in enumerate(relevant_flags, 1):
        if rel:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def context_recall(reference: str, contexts: list[str], tau: float = 0.6) -> tuple[float, list[Support]]:
    sup = [support(s, contexts) for s in sentences(reference)]
    return (sum(s.ratio >= tau for s in sup) / len(sup) if sup else 0.0), sup


def preset_answers(contexts: list[str], domain: str, other_domain_text: str) -> dict[str, str]:
    first = sentences(contexts[0])[0] if contexts else ""
    return {
        "Grounded (copied from the top context)": first,
        "Partly made up (grounded + an invented claim)": (first + " " + FABRICATED[domain]).strip(),
        "Off-topic (from another domain)": other_domain_text,
    }


if __name__ == "__main__":
    ctx = ["Most adults need seven to nine hours of sleep per night; regularly sleeping less than seven hours is "
           "associated with weight gain, high blood pressure and poorer concentration."]
    ans = "Adults need seven to nine hours of sleep. " + FABRICATED["Health & nutrition"]
    f, sup = faithfulness(ans, ctx)
    print("faithfulness", f, [(s.sentence[:40], round(s.ratio, 2)) for s in sup])
    print("context precision", context_precision([True, False, True]))
