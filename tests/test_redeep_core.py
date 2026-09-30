import json
import tempfile
import unittest
from pathlib import Path

from src.hallucination_labels import retrieval_sufficient
from src.prompt_utils import (
    TRUNCATION_MARKER,
    decode_generated_response,
    fit_prompt_parts,
    normalize_generated_token_ids,
    token_length,
)
from src.redeep.calibration import ReDeEPCalibrator
from src.redeep.scores import stable_jsd
from src.redeep.io import build_prompt_parts, enrich_record, read_records


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=True):
        prefix = [0] if add_special_tokens else []
        return {"input_ids": prefix + list(range(len(text)))}


class DecodeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def decode(self, token_ids, **kwargs):
        return "decoded answer"


class ReDeEPCoreTests(unittest.TestCase):
    def test_prompt_truncation_preserves_both_ends_and_exact_parts(self):
        tokenizer = CharacterTokenizer()
        context = "first evidence " + ("middle " * 20) + "last evidence"
        parts = fit_prompt_parts(tokenizer, "P:", context, ":S", max_tokens=60, response="answer")
        self.assertTrue(parts["context_truncated"])
        self.assertIn(TRUNCATION_MARKER.strip(), parts["context"])
        self.assertTrue(parts["context"].startswith("first"))
        self.assertTrue(parts["context"].endswith("evidence"))
        self.assertEqual(parts["prompt"], parts["prefix"] + parts["context"] + parts["suffix"])
        self.assertLessEqual(token_length(tokenizer, parts["prompt"] + "answer"), 60)

    def test_generated_token_ids_drop_eos_and_flashrag_padding(self):
        self.assertEqual(normalize_generated_token_ids([[10, 11, 2, 0, 0]], eos_token_id=2, pad_token_id=0), [10, 11])
        self.assertEqual(normalize_generated_token_ids([10, "bad"], eos_token_id=2), [])

    def test_flashrag_answer_uses_exact_generated_token_ids(self):
        self.assertEqual(
            decode_generated_response(DecodeTokenizer(), [10, 11], "possibly mis-sliced"), "decoded answer"
        )
        self.assertEqual(decode_generated_response(DecodeTokenizer(), [], "fallback answer"), "fallback answer")

    def test_saved_prompt_parts_and_hop_count_are_preserved(self):
        parts = {
            "prompt": "prefixcontextsuffix",
            "prefix": "prefix",
            "context": "context",
            "suffix": "suffix",
            "context_truncated": True,
        }
        record = {
            "id": "x",
            "question": "q",
            "prediction": "a",
            "response_token_ids": [10, 11],
            "golden_answers": ["a"],
            "hop_num": 4,
            "trace": {"hop_num": 4, "hops": []},
            "prompt_parts": parts,
        }
        enriched = enrich_record(record)
        self.assertEqual(enriched["prompt_parts"], parts)
        self.assertEqual(enriched["hop_num"], 4)
        self.assertEqual(enriched["response_token_ids"], [10, 11])

    def test_invalid_saved_prompt_parts_are_rejected(self):
        record = {
            "question": "q",
            "prediction": "a",
            "golden_answers": ["a"],
            "trace": {},
            "prompt_parts": {"prompt": "wrong", "prefix": "p", "context": "c", "suffix": "s"},
        }
        with self.assertRaisesRegex(ValueError, "do not reconstruct"):
            enrich_record(record)

    def test_saved_answer_prompt_must_match_trace_evidence(self):
        trace = {"hops": [{"hop": 1, "documents": [{"title": "Evidence", "text": "text"}]}]}
        with self.assertRaisesRegex(ValueError, "does not contain"):
            build_prompt_parts("q", trace, answer_prompt="Answer using unrelated evidence")

    def test_flashrag_contents_title_supports_retrieval_aware_mode(self):
        record = {"metadata": {"supporting_facts": {"title": ["Target Page"]}}}
        trace = {"hops": [{"documents": [{"id": "10", "contents": "Target Page\nArticle text"}]}]}
        self.assertTrue(retrieval_sufficient(record, trace))

    def test_official_flashrag_saved_layout_is_read(self):
        payload = [
            {
                "id": "1",
                "question": "q",
                "golden_answers": ["a"],
                "output": {
                    "pred": "a",
                    "redeep_records": {
                        "id": "1",
                        "question": "q",
                        "gold_answers": ["a"],
                        "prediction": "a",
                        "trace": {"hops": []},
                    },
                },
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records = read_records(str(path))
        self.assertEqual(records[0]["prediction"], "a")
        self.assertEqual(records[0]["gold_answers"], ["a"])

    def test_standard_jsd_is_symmetric_and_zero_for_equal_logits(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is installed in the GPU runtime, not this lightweight local test environment")

        left = torch.tensor([[2.0, 0.0, -1.0]])
        right = torch.tensor([[-1.0, 0.5, 2.0]])
        self.assertAlmostEqual(float(stable_jsd(left, left).item()), 0.0, places=7)
        self.assertAlmostEqual(
            float(stable_jsd(left, right).item()),
            float(stable_jsd(right, left).item()),
            places=7,
        )

    def test_joint_calibration_fit_and_round_trip(self):
        records = []
        for index in range(6):
            label = index % 2
            records.append(
                {
                    "hallucination_label": label,
                    "ecs": {"layer_0_head_0": [0.9 if label == 0 else 0.1]},
                    "pks": {"0": [0.1 if label == 0 else 0.9]},
                }
            )
        calibrator = ReDeEPCalibrator(top_heads=1, top_layers=1).fit(records)
        self.assertEqual(calibrator.selected_heads, ["layer_0_head_0"])
        self.assertEqual(calibrator.selected_layers, ["0"])
        self.assertGreater(calibrator.score(records[1]), calibrator.score(records[0]))
        calibrator.runtime_settings = {
            "model": "model/Qwen3-4B-Instruct-2507",
            "granularity": "token",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibrator.save(str(path))
            loaded = ReDeEPCalibrator.load(str(path))
        self.assertEqual(loaded.to_dict(), calibrator.to_dict())
        loaded.validate_runtime({"model": "model/Qwen3-4B-Instruct-2507", "granularity": "token"})
        with self.assertRaisesRegex(ValueError, "granularity"):
            loaded.validate_runtime({"model": "model/Qwen3-4B-Instruct-2507", "granularity": "chunk"})
        with self.assertRaisesRegex(ValueError, "missing calibrated"):
            loaded.score({"ecs": {}, "pks": {"0": [0.5]}})

    def test_paper_top_k_search_clips_to_available_features(self):
        records = []
        for index in range(20):
            label = index % 2
            records.append(
                {
                    "hallucination_label": label,
                    "ecs": {
                        "layer_0_head_0": 0.9 if label == 0 else 0.1,
                        "layer_0_head_1": 0.8 if label == 0 else 0.2,
                    },
                    "pks": {
                        "0": 0.1 if label == 0 else 0.9,
                        "1": 0.2 if label == 0 else 0.8,
                    },
                }
            )
        calibrator = ReDeEPCalibrator(top_heads=32, top_layers=32).fit(
            records[:16], validation_records=records[16:]
        )
        self.assertGreaterEqual(len(calibrator.selected_heads), 1)
        self.assertLessEqual(len(calibrator.selected_heads), 2)
        self.assertGreaterEqual(len(calibrator.selected_layers), 1)
        self.assertLessEqual(len(calibrator.selected_layers), 2)
        self.assertIn(calibrator.alpha, [value / 10.0 for value in range(1, 20)])


if __name__ == "__main__":
    unittest.main()
