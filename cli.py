#!/usr/bin/env python3
"""Command-line entrypoint for the Delta support agent.

Usage:
    python cli.py setup                 # (re)generate sample data + golden sets
    python cli.py handle "your message" # classify + reply + escalation for one msg
    python cli.py demo                  # run a curated set of example messages
    python cli.py eval                  # run the full evaluation harness

Environment:
    TWCS_PATH=/path/to/twcs.csv         # point at the real Kaggle file
    SUPPORT_AGENT_LLM=openai|anthropic  # use a hosted model (needs API key+net)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from support_agent.agent import SupportAgent
from support_agent.config import Config


DEMO_MESSAGES = [
    "@Delta my flight got cancelled and I'm stuck at JFK, need to rebook now!!",
    "@Delta you charged me twice for my bag, I want a refund. calling my lawyer",
    "@Delta my bag never showed up at LAX, where is it??",
    "@Delta huge thanks to the crew today, amazing service ✈️",
    "@Delta can I bring my dog in the cabin on a flight to Boston?",
    "@Delta someone used my SkyMiles to book a flight I didn't authorize",
    "@Delta boarding in 5 minutes and my mobile pass just disappeared!!!",
    "@Delta the gate agent was openly rude to a disabled passenger. disgraceful",
    "@Delta 🔥🔥🔥",
]


def _print_response(resp):
    print("\n" + "-" * 72)
    print("IN : " + resp.text_in)
    flag = "ESCALATE" if resp.escalate else "AUTO-HANDLE"
    print(f"INTENT: {resp.intent}  (conf={resp.confidence:.2f})   ROUTE: {flag}")
    print(f"WHY : {resp.reason}")
    print(f"REPLY: {resp.reply}")
    print(f"GROUNDED: {resp.reply_grounded}", end="")
    if resp.evidence:
        top = resp.evidence[0]
        print(f"  | top evidence (sim={top['similarity']}): \"{top['past_resolution'][:70]}...\"")
    else:
        print()


def cmd_handle(cfg, text):
    agent = SupportAgent(cfg).build()
    _print_response(agent.handle(text))


def cmd_demo(cfg):
    agent = SupportAgent(cfg).build()
    for m in DEMO_MESSAGES:
        _print_response(agent.handle(m))


def cmd_setup(cfg):
    import subprocess
    here = os.path.dirname(__file__)
    for script in ["scripts/generate_sample_data.py", "scripts/build_golden.py",
                   "scripts/build_judge_set.py"]:
        print(f"$ python {script}")
        subprocess.run([sys.executable, os.path.join(here, script)], check=True, cwd=here)


def cmd_eval(cfg):
    import subprocess
    here = os.path.dirname(__file__)
    subprocess.run([sys.executable, os.path.join(here, "eval/run_eval.py")], check=True, cwd=here)


def main():
    cfg = Config()
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "handle":
        if len(sys.argv) < 3:
            print("usage: python cli.py handle \"message\"")
            return
        cmd_handle(cfg, sys.argv[2])
    elif cmd == "demo":
        cmd_demo(cfg)
    elif cmd == "setup":
        cmd_setup(cfg)
    elif cmd == "eval":
        cmd_eval(cfg)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
