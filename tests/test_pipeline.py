"""Core invariant tests (stdlib unittest, no deps).

Run: python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from support_agent.config import Config
from support_agent.agent import SupportAgent
from support_agent.data_loader import build_threads
from support_agent.escalation import decide
from support_agent.classifiers.base import Prediction
from support_agent.intents import INTENTS, ABSTAIN
from support_agent.text import tokenize, normalize
from support_agent.vectorizer import TfidfVectorizer, cosine
from eval import metrics as M


class TestText(unittest.TestCase):
    def test_normalize_masks_pii_like_tokens(self):
        n = normalize("@Delta see http://x.co now 12345")
        self.assertIn("<user>", n)
        self.assertIn("<url>", n)
        self.assertIn("<num>", n)

    def test_tokenize_drops_stopwords(self):
        toks = tokenize("I am at the gate")
        self.assertNotIn("the", toks)
        self.assertIn("gate", toks)


class TestVectorizer(unittest.TestCase):
    def test_cosine_self_is_one(self):
        v = TfidfVectorizer(min_df=1).fit(["lost my bag", "flight delayed"])
        a = v.transform("lost my bag")
        self.assertAlmostEqual(cosine(a, a), 1.0, places=5)

    def test_cosine_disjoint_is_zero(self):
        v = TfidfVectorizer(min_df=1).fit(["alpha beta", "gamma delta"])
        self.assertEqual(cosine(v.transform("alpha beta"), v.transform("gamma delta")), 0.0)


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(M.accuracy(["a", "b", "c"], ["a", "b", "x"]), 2 / 3)

    def test_cohen_kappa_perfect(self):
        self.assertAlmostEqual(M.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)

    def test_cohen_kappa_chance(self):
        # identical marginals, all agreement by chance -> kappa ~ 0
        self.assertAlmostEqual(M.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)
        k = M.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1])
        self.assertLess(k, 0.0)


class TestEscalation(unittest.TestCase):
    def test_pii_forces_escalation(self):
        pred = Prediction(label="general_info", proba={"general_info": 0.99})
        d = decide("email me at a@b.com about my trip", pred)
        self.assertTrue(d.escalate)
        self.assertIn("pii_exposed_needs_private_channel", d.signals)

    def test_human_request_escalates(self):
        pred = Prediction(label="praise", proba={"praise": 0.99})
        d = decide("please connect me to a real person", pred)
        self.assertTrue(d.escalate)

    def test_confident_lowstakes_autohandles(self):
        pred = Prediction(label="general_info",
                          proba={"general_info": 0.95, "baggage": 0.05})
        d = decide("what is the carry on size limit", pred)
        self.assertFalse(d.escalate)

    def test_stem_regex_bug_fixed(self):
        # "humiliating" must trigger the distress signal (regressive bug)
        pred = Prediction(label="praise", proba={"praise": 0.9, "baggage": 0.1})
        d = decide("this was utterly humiliating", pred)
        self.assertTrue(d.escalate)


class TestAgentEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        # ensure sample data exists
        if not os.path.exists(os.path.join(ROOT, cls.cfg.data_path)):
            from scripts.generate_sample_data import generate
            generate(500, 13, os.path.join(ROOT, cls.cfg.data_path))
        os.chdir(ROOT)
        cls.agent = SupportAgent(cls.cfg).build()

    def test_threads_reconstructed(self):
        threads = build_threads(self.cfg.data_path, self.cfg.brand)
        self.assertGreater(len(threads), 100)
        self.assertTrue(all(t.is_resolved for t in threads))

    def test_handle_returns_valid_intent(self):
        r = self.agent.handle("@Delta my flight got cancelled, need to rebook")
        self.assertIn(r.intent, set(INTENTS) | {ABSTAIN})
        self.assertTrue(r.escalate)  # disruption is high-stakes
        self.assertGreater(len(r.reply), 10)

    def test_reply_never_solicits_card_or_password(self):
        for msg in ["@Delta refund me", "@Delta cant log in", "@Delta lost bag"]:
            r = self.agent.handle(msg)
            low = r.reply.lower()
            self.assertNotIn("card number", low)
            self.assertNotIn("password", low)
            self.assertNotIn("cvv", low)


if __name__ == "__main__":
    unittest.main()
