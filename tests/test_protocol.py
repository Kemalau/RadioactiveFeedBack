import json
from pathlib import Path
import tempfile
import unittest

from radioactive_feedback.data import prepare_tasks, task_batches
from radioactive_feedback.judge import parse_scores
from radioactive_feedback.provider import Carrier, PromptConfig, build_system_prompt


class ProtocolTests(unittest.TestCase):
    def test_no_train_audit_leakage_and_last_partial_batch(self):
        tasks = [{"task_id": "a", "prompt": "Two  plus two"},
                 {"task_id": "b", "prompt": "Two plus\ntwo"},
                 {"task_id": "c", "prompt": "Three plus four"},
                 {"task_id": "d", "prompt": "Five plus six"}]
        retained, metadata = prepare_tasks(tasks, [{"task_id": "audit", "prompt": "Three plus  four"}], 42)
        self.assertEqual(metadata["removed_train_duplicates"], 1)
        self.assertEqual(metadata["removed_audit_overlap"], 1)
        self.assertEqual(retained, prepare_tasks(tasks, [{"task_id": "audit", "prompt": "Three plus four"}], 42)[0])
        self.assertEqual([len(batch) for batch in task_batches(retained, 50)], [2])

    def test_scores_fail_closed(self):
        self.assertEqual(parse_scores('{"scores":[0, 99.5, 100]}', 3), [0, 99.5, 100])
        self.assertEqual(parse_scores('{"score":90}', 1), [90])
        for raw, count in [('{"scores":[true]}', 1), ('{"scores":[101]}', 1),
                           ('{"scores":[NaN]}', 1), ('{"scores":[90]}', 2),
                           ('{"scores":[90],"reasoning":"secret"}', 1), ('90', 1)]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_scores(raw, count)

    def test_explicit_preference_policy(self):
        policy = PromptConfig(carrier_pool=(
            Carrier("test_carrier", "Test domain", "Test preferred and opposite alternatives"),))
        prompt = build_system_prompt(policy)
        self.assertIn(policy.selected[0].rule, prompt)
        self.assertNotIn("c_j(x)", prompt)
        self.assertIn("delta_i = 5", prompt)
        self.assertIn("r_min=60", prompt)
        self.assertIn("does not require a near tie", prompt)

    def test_policy_matches_provider_package(self):
        try:
            from keyflip_api.prompt import Carrier as APICarrier, PromptConfig as APIConfig
            from keyflip_api.prompt import build_system_prompt as api_prompt
        except ImportError:
            self.skipTest("provider package not installed")
        carrier = Carrier("test_carrier", "Test domain", "Test private preference")
        policy = PromptConfig(carrier_pool=(carrier,))
        api_policy = APIConfig(carrier_pool=(APICarrier(
            carrier.identifier, carrier.domain, carrier.rule),))
        self.assertEqual(build_system_prompt(policy), api_prompt(api_policy))


if __name__ == "__main__":
    unittest.main()
