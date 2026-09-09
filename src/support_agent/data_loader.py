"""Load the Kaggle "Customer Support on Twitter" CSV and reconstruct threads.

Real schema (thoughtvector/customer-support-on-twitter, file `twcs.csv`):
    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

`inbound == "True"` means the tweet is FROM a customer TO a brand.
Brand agents tweet with `inbound == "False"` and an `author_id` equal to the
brand handle (e.g. "Delta", "AppleSupport").

This module is schema-faithful: it runs on the full Kaggle file unchanged and on
the bundled sample corpus identically. For large files we stream row by row.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass
class Tweet:
    tweet_id: str
    author_id: str
    inbound: bool
    created_at: str
    text: str
    response_tweet_id: str = ""
    in_response_to_tweet_id: str = ""


@dataclass
class Thread:
    """A reconstructed conversation anchored on a customer's first inbound tweet."""
    root_id: str
    brand: str
    customer_text: str            # first inbound customer message (the "ask")
    tweets: List[Tweet]           # full ordered thread

    @property
    def agent_replies(self) -> List[Tweet]:
        return [t for t in self.tweets if not t.inbound and t.author_id.lower() == self.brand.lower()]

    @property
    def resolution_text(self) -> str:
        """Concatenated brand-agent replies — the historical 'how we resolved it'."""
        return " ".join(t.text for t in self.agent_replies).strip()

    @property
    def is_resolved(self) -> bool:
        return len(self.agent_replies) > 0


def _to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "t")


def read_tweets(path: str) -> Dict[str, Tweet]:
    """Stream the CSV into an id->Tweet map. Tolerant of column-order variation."""
    tweets: Dict[str, Tweet] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row.get("tweet_id", "")).strip()
            if not tid:
                continue
            tweets[tid] = Tweet(
                tweet_id=tid,
                author_id=str(row.get("author_id", "")).strip(),
                inbound=_to_bool(row.get("inbound", "")),
                created_at=str(row.get("created_at", "")).strip(),
                text=str(row.get("text", "")),
                response_tweet_id=str(row.get("response_tweet_id", "")).strip(),
                in_response_to_tweet_id=str(row.get("in_response_to_tweet_id", "")).strip(),
            )
    return tweets


def _follow_thread(root: Tweet, tweets: Dict[str, Tweet]) -> List[Tweet]:
    """Walk response_tweet_id links forward from the root."""
    chain: List[Tweet] = [root]
    seen = {root.tweet_id}
    cur = root
    while cur.response_tweet_id:
        # response_tweet_id may be a comma-separated list; take the first unseen.
        next_id = None
        for cand in cur.response_tweet_id.split(","):
            cand = cand.strip()
            if cand and cand in tweets and cand not in seen:
                next_id = cand
                break
        if not next_id:
            break
        cur = tweets[next_id]
        chain.append(cur)
        seen.add(cur.tweet_id)
    return chain


def build_threads(path: str, brand: str) -> List[Thread]:
    """Reconstruct threads whose customer is talking to `brand`.

    A thread qualifies if either (a) the root inbound tweet is a reply that a
    brand agent responded to, or (b) the brand handle appears as a responder in
    the chain. We anchor on the *first inbound customer tweet* that has no
    inbound parent (the original ask).
    """
    tweets = read_tweets(path)
    brand_l = brand.lower()

    # Identify roots: inbound tweets not themselves in_response_to another inbound.
    roots: List[Tweet] = []
    for t in tweets.values():
        if not t.inbound:
            continue
        parent = tweets.get(t.in_response_to_tweet_id)
        if parent is not None and parent.inbound:
            continue  # not the first ask in the thread
        roots.append(t)

    threads: List[Thread] = []
    for root in roots:
        chain = _follow_thread(root, tweets)
        # Does a brand agent participate?
        if not any((not x.inbound and x.author_id.lower() == brand_l) for x in chain):
            continue
        threads.append(
            Thread(
                root_id=root.tweet_id,
                brand=brand,
                customer_text=root.text,
                tweets=chain,
            )
        )
    # stable order for reproducibility
    threads.sort(key=lambda th: th.root_id)
    return threads


def iter_inbound_for_brand(path: str, brand: str) -> Iterator[Tweet]:
    """Yield inbound customer tweets addressed to the brand (mentions the handle)."""
    brand_at = "@" + brand.lower()
    for t in read_tweets(path).values():
        if t.inbound and brand_at in t.text.lower():
            yield t
