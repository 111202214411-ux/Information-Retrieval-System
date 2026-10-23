"""
subword.py: subword tokenization from scratch.

- BPE (Byte Pair Encoding, Sennrich et al. 2016): start from characters and repeatedly
  merge the most FREQUENT adjacent pair.
- WordPiece (used by BERT): same loop, but merge the pair with the highest
  LIKELIHOOD score  freq(ab) / (freq(a) * freq(b)), and mark word-internal pieces with '##'.

Both follow the lecture pseudocode:

    FUNCTION BPE_Train(corpus, num_merges):
      vocab = all unique characters
      tokens = split words into chars
      FOR i = 1 TO num_merges:
        pairs = count_adjacent_pairs(tokens)
        best = argmax(pairs, by frequency)
        new_token = merge(best)
        vocab.add(new_token)
        tokens = replace_all(tokens, best, new_token)
      RETURN vocab, merge_rules

Try it from the terminal:
    python subword.py
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

END = "</w>"          # end-of-word marker used by the original BPE paper
UNK = "[UNK]"

# Ready-made training corpora: word -> frequency
PRESET_CORPORA = {
    "Sennrich et al. (2016): low, lower, newest, widest":
        {"low": 5, "lower": 2, "newest": 6, "widest": 3},
    "Hugging Face course: hug, pug, pun, bun, hugs":
        {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5},
    "Indonesian affixes: main, bermain, permainan, ...":
        {"main": 6, "bermain": 5, "permainan": 4, "dimainkan": 3, "memainkan": 3, "pemain": 4,
         "ajar": 4, "belajar": 6, "pelajaran": 5, "pembelajaran": 3, "mengajar": 3},
}


def word_frequencies(text: str) -> dict[str, int]:
    """Build a training corpus (word -> count) from free text: lowercase letters only."""
    return dict(Counter(re.findall(r"[^\W\d_]+", text.lower())))


def parse_corpus(spec: str) -> dict[str, int]:
    """
    Parse 'word count' lines (e.g. 'low 5'). Lines without a count are treated as free text.
    """
    freqs: Counter = Counter()
    for line in spec.splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) == 2 and parts[1].isdigit():
            freqs[parts[0].lower()] += int(parts[1])
        elif parts:
            freqs.update(word_frequencies(line))
    return dict(freqs)


@dataclass
class MergeStep:
    step: int
    pair: tuple[str, str]
    score: float              # frequency for BPE, likelihood score for WordPiece
    new_token: str
    vocab_size: int


@dataclass
class Model:
    kind: str                                  # "bpe" | "wordpiece"
    base_vocab: list[str]
    merges: list[MergeStep]
    splits: dict[str, list[str]]               # how each training word is segmented at the end
    vocab: list[str] = field(default_factory=list)

    def tokenize(self, word: str) -> list[str]:
        word = word.lower()
        return bpe_tokenize(word, self) if self.kind == "bpe" else wordpiece_tokenize(word, self.vocab)


def _merge_pair(pieces: list[str], a: str, b: str, new: str) -> list[str]:
    """Replace every non-overlapping occurrence of the adjacent pair (a, b), left to right."""
    out, i = [], 0
    while i < len(pieces):
        if i + 1 < len(pieces) and pieces[i] == a and pieces[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
# BPE
# ---------------------------------------------------------------------------

def bpe_train(freqs: dict[str, int], num_merges: int) -> Model:
    splits = {w: list(w) + [END] for w in freqs}
    base = sorted({p for s in splits.values() for p in s})
    vocab = list(base)
    merges: list[MergeStep] = []

    for step in range(1, num_merges + 1):
        pairs: Counter = Counter()
        for w, f in freqs.items():
            s = splits[w]
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        (a, b), freq = max(pairs.items(), key=lambda kv: kv[1])   # ties: first pair seen wins
        new = a + b
        splits = {w: _merge_pair(s, a, b, new) for w, s in splits.items()}
        if new not in vocab:
            vocab.append(new)
        merges.append(MergeStep(step, (a, b), freq, new, len(vocab)))

    return Model("bpe", base, merges, splits, vocab)


def bpe_tokenize(word: str, model: Model) -> list[str]:
    """Split into characters, then apply the learned merge rules in the order they were learned."""
    pieces = list(word) + [END]
    for m in model.merges:
        pieces = _merge_pair(pieces, *m.pair, m.new_token)
    return pieces


# ---------------------------------------------------------------------------
# WordPiece
# ---------------------------------------------------------------------------

def _wp_split(word: str) -> list[str]:
    return [word[0]] + ["##" + c for c in word[1:]]


def wordpiece_train(freqs: dict[str, int], num_merges: int) -> Model:
    splits = {w: _wp_split(w) for w in freqs}
    base = sorted({p for s in splits.values() for p in s}, key=lambda p: (p.startswith("##"), p))
    vocab = list(base)
    merges: list[MergeStep] = []

    for step in range(1, num_merges + 1):
        piece_freq: Counter = Counter()
        pair_freq: Counter = Counter()
        for w, f in freqs.items():
            s = splits[w]
            for p in s:
                piece_freq[p] += f
            for a, b in zip(s, s[1:]):
                pair_freq[(a, b)] += f
        if not pair_freq:
            break
        scores = {pr: c / (piece_freq[pr[0]] * piece_freq[pr[1]]) for pr, c in pair_freq.items()}
        (a, b), score = max(scores.items(), key=lambda kv: kv[1])
        new = a + (b[2:] if b.startswith("##") else b)
        splits = {w: _merge_pair(s, a, b, new) for w, s in splits.items()}
        if new not in vocab:
            vocab.append(new)
        merges.append(MergeStep(step, (a, b), score, new, len(vocab)))

    return Model("wordpiece", base, merges, splits, vocab)


def wordpiece_tokenize(word: str, vocab: list[str]) -> list[str]:
    """Greedy longest-match-first, as in BERT. A word that cannot be covered becomes [UNK]."""
    known, tokens, start = set(vocab), [], 0
    while start < len(word):
        end, piece = len(word), None
        while start < end:
            sub = word[start:end] if start == 0 else "##" + word[start:end]
            if sub in known:
                piece = sub
                break
            end -= 1
        if piece is None:
            return [UNK]
        tokens.append(piece)
        start = end
    return tokens


def word_level_tokenize(word: str, freqs: dict[str, int]) -> list[str]:
    """Word-level tokenization: a word is either in the vocabulary or it is out-of-vocabulary (OOV)."""
    return [word.lower()] if word.lower() in freqs else [UNK]


# ---------------------------------------------------------------------------
# Pretrained tokenizers from the Hugging Face Hub (optional, needs internet)
# ---------------------------------------------------------------------------

PRETRAINED = {
    "bert-base-uncased": ("BERT", "WordPiece"),
    "gpt2": ("GPT-2", "Byte-level BPE"),
    "indobenchmark/indobert-base-p1": ("IndoBERT", "WordPiece"),
}


def load_pretrained(name: str):
    """
    Download a tokenizer from the Hugging Face Hub with the lightweight `tokenizers` library
    (no PyTorch or transformers needed). Raises an exception if the Hub cannot be reached.
    """
    from tokenizers import Tokenizer
    try:
        return Tokenizer.from_pretrained(name)
    except Exception:
        # some repositories only ship vocab.txt (no tokenizer.json): build a BERT WordPiece tokenizer from it
        from huggingface_hub import hf_hub_download
        from tokenizers.implementations import BertWordPieceTokenizer
        return BertWordPieceTokenizer(hf_hub_download(name, "vocab.txt"), lowercase=True)


if __name__ == "__main__":
    corpus = PRESET_CORPORA["Sennrich et al. (2016): low, lower, newest, widest"]
    bpe = bpe_train(corpus, 10)
    print("BPE merges:")
    for m in bpe.merges:
        print(f"  {m.step:>2}. {m.pair[0]} + {m.pair[1]} -> {m.new_token}   (freq {m.score:g})")
    wp = wordpiece_train(corpus, 10)
    print("WordPiece merges:")
    for m in wp.merges:
        print(f"  {m.step:>2}. {m.pair[0]} + {m.pair[1]} -> {m.new_token}   (score {m.score:.4f})")
    for w in ["lowest", "newer", "wider", "slow"]:
        print(f"{w:>8}: word-level {word_level_tokenize(w, corpus)}  BPE {bpe.tokenize(w)}  WordPiece {wp.tokenize(w)}")
