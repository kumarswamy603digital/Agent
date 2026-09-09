"""Pluggable LLM backend.

The whole agent talks to an LLM only through the `LLMBackend.complete(system,
user)` interface. This lets us run three ways with zero code changes elsewhere:

  * HeuristicBackend  -> deterministic, no API key required. DEFAULT. Used for
                         stable, reproducible headline numbers.
  * OpenAIBackend     -> hosted model via HTTPS (needs OPENAI_API_KEY).
  * AnthropicBackend  -> hosted model via HTTPS (needs ANTHROPIC_API_KEY).

The default backend keeps evaluation deterministic and dependency-free. The API
adapters are implemented with the standard-library `urllib`, so setting
`SUPPORT_AGENT_LLM=openai` (with a key) yields true generative replies and judging
without any other code change.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional


class LLMBackend:
    name = "base"

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        return True


# --------------------------------------------------------------------------- #
# Default deterministic backend                                               #
# --------------------------------------------------------------------------- #
class HeuristicBackend(LLMBackend):
    """Deterministic text synthesis.

    It is NOT a language model. It performs two jobs the rest of the code asks
    of an LLM, using structured prompts we control:

      1. Reply drafting: the prompt embeds retrieved exemplar resolutions and a
         template scaffold; this backend extracts the scaffold and fills it. In
         practice the reply.py module does the heavy lifting and only uses this
         for light smoothing, so behavior is fully reproducible.
      2. Judging: returns a JSON verdict computed by a rubric heuristic
         (see judge.py, which calls back into deterministic scorers).

    Because it's deterministic, metrics are stable run-to-run.
    """

    name = "heuristic"

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        # The caller passes a JSON control block after a marker when it wants a
        # structured operation. Otherwise we echo a safe canned scaffold.
        m = re.search(r"<<CONTROL>>(.*)<<END>>", user, re.S)
        if not m:
            return user.strip()[:max_tokens]
        try:
            ctrl = json.loads(m.group(1))
        except Exception:
            return user.strip()[:max_tokens]
        # We simply return the pre-rendered draft the caller assembled; the
        # backend boundary exists so a real LLM can replace this transparently.
        return ctrl.get("draft", "").strip()


# --------------------------------------------------------------------------- #
# Hosted API adapters (used only when network + key are present)              #
# --------------------------------------------------------------------------- #
class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: str = ""):
        self.model = model or os.environ.get("SUPPORT_AGENT_LLM_MODEL", "gpt-4o-mini")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        import urllib.request

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                }
            ).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()


class AnthropicBackend(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str = ""):
        self.model = model or os.environ.get("SUPPORT_AGENT_LLM_MODEL", "claude-3-5-sonnet-latest")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        import urllib.request

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(
                {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                }
            ).encode(),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        return data["content"][0]["text"].strip()


def get_backend(name: Optional[str] = None, model: str = "") -> LLMBackend:
    name = (name or os.environ.get("SUPPORT_AGENT_LLM", "heuristic")).lower()
    if name == "openai":
        b = OpenAIBackend(model)
        if b.available():
            return b
    if name == "anthropic":
        b = AnthropicBackend(model)
        if b.available():
            return b
    return HeuristicBackend()
