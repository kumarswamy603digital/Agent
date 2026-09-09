"""Text normalization and tokenization for noisy Twitter text (stdlib only)."""

from __future__ import annotations

import re
import unicodedata
from typing import List

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_NUM_RE = re.compile(r"\b\d[\d,]*\b")
_WORD_RE = re.compile(r"[a-z0-9']+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")  # loooove -> loove

# A tiny stopword list. Kept short on purpose: aggressive stopword removal hurts
# short-text intent signal (e.g. "where is" matters for baggage tracking).
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "am", "it", "its", "this", "that",
    "i", "im", "my", "me", "we", "you", "your", "u", "at", "as", "so", "with",
}

# Anonymization tokens keep semantics while removing PII/handles/urls.
TOK_URL = "<url>"
TOK_USER = "<user>"
TOK_NUM = "<num>"


def normalize(text: str) -> str:
    """Lowercase, strip accents, collapse urls/mentions/numbers, de-elongate."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = _URL_RE.sub(f" {TOK_URL} ", text)
    text = _MENTION_RE.sub(f" {TOK_USER} ", text)
    text = _HASHTAG_RE.sub(r" \1 ", text)  # keep hashtag word, drop the '#'
    text = _NUM_RE.sub(f" {TOK_NUM} ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    return text


def tokenize(text: str, keep_stop: bool = False) -> List[str]:
    """Return unigram tokens; special <...> tokens are preserved intact."""
    norm = normalize(text)
    toks: List[str] = []
    for m in re.finditer(r"<\w+>|[a-z0-9']+", norm):
        t = m.group(0)
        if not keep_stop and t in STOPWORDS:
            continue
        toks.append(t)
    return toks


def ngrams(tokens: List[str], n: int) -> List[str]:
    if n <= 1:
        return list(tokens)
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def featurize(text: str, ngram_range=(1, 2)) -> List[str]:
    """Bag of uni+bi-gram string features used by the vectorizer/NB."""
    toks = tokenize(text)
    feats: List[str] = []
    lo, hi = ngram_range
    for n in range(lo, hi + 1):
        feats.extend(ngrams(toks, n))
    return feats
