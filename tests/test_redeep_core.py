import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.hallucination_labels import LABEL_RULE_VERSION, retrieval_sufficient
from src.prompt_utils import (
    TRUNCATION_MARKER,
    decode_generated_response,
    fit_prompt_parts,
    normalize_generated_token_ids,
    token_length,
)
from src.redeep.calibration import ReDeEPCalibrator
from src.redeep.detector import ReDeEPDetector
from src.redeep.scores import stable_jsd
from src.redeep.io import build_prompt_parts, enrich_record, read_records
from src.run_redeep import _metrics, _parser, _runtime_settings, main as run_redeep


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
    def test_fit_and_evaluate_relabel_raw_answers_and_keep_their_token_ids(self):
        cases = [
            ('Yes, it is.', ['yes'], 0), ('Yes, no doubt.', ['yes'], 1),
            ('PARIS.', ['Paris'], 0), ('London', ['Paris'], 1),
            ('3,677', ['3,677 seated'], 0), ('No, it is not.', ['yes'], 1),
        ]
        records = [
            {'id': str(i), 'question': 'Q', 'prediction': response, 'gold_answers': golds,
             'response_token_ids': [100 + i, 200 + i], 'hallucination_label': 0,
             'trace': {'hop_num': 2, 'hops': [{'hop': 1, 'documents': [{'doc_id': 'd', 'text': 'Evidence'}]}]}}
            for i, (response, golds, _) in enumerate(cases)
        ]
        captured_batches = []

        def make_detector(**kwargs):
            detector = object.__new__(ReDeEPDetector)
            detector.max_input_tokens = None
            detector.calibrator = kwargs.get('calibrator')

            def extract_batch(requests):
                captured_batches.append(requests)
                return [object() for _ in requests]

            detector.extractor = SimpleNamespace(extract_batch_with_parts=extract_batch)

            def score_features(record, representation, include_token_scores):
                # Fixed stand-in features exercise real labeling, batching,
                # calibration and file IO without needing Qwen or Torch.
                feature = 0.1 + 0.1 * int(record['id'])
                scored = {**record, 'ecs': {'layer_0_head_0': 1.0 - feature}, 'pks': {'0': feature}}
                scored['redeep_score'] = detector.calibrator.score(scored) if detector.calibrator else None
                scored['redeep_prediction'] = detector.calibrator.predict(scored) if detector.calibrator else None
                return scored

            detector._score_representation = Mock(side_effect=score_features)
            return detector

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.jsonl'
            source.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
            calibration = root / 'fit.calibration.json'
            for command in ['fit', 'evaluate']:
                output = root / (command + '.jsonl')
                argv = ['run_redeep', command, '--input', str(source), '--output', str(output),
                        '--calibration', str(calibration), '--top-heads', '1', '--top-layers', '1',
                        '--batch-size', '2', '--no-token-scores']
                with patch('sys.argv', argv), patch('src.run_redeep.ReDeEPDetector', side_effect=make_detector), redirect_stdout(io.StringIO()):
                    run_redeep()
                scored = read_records(str(output))
                self.assertEqual(len(scored), len(cases))
                for record, original, (_, _, expected) in zip(scored, records, cases):
                    self.assertEqual(record['hallucination_label'], expected)
                    self.assertEqual(record['hallucination_label_version'], LABEL_RULE_VERSION)
                    self.assertEqual(record['prediction'], original['prediction'])
                    self.assertEqual(record['response_token_ids'], original['response_token_ids'])
            saved = json.loads(calibration.read_text(encoding='utf-8'))
            self.assertEqual(saved['runtime_settings']['label_rule_version'], LABEL_RULE_VERSION)
        self.assertEqual([len(batch) for batch in captured_batches], [2, 2, 2, 2, 2, 2])
        for requests, originals in zip(captured_batches, [records[0:2], records[2:4], records[4:6]] * 2):
            self.assertEqual([item['response'] for item in requests], [item['prediction'] for item in originals])
            self.assertEqual([item['response_token_ids'] for item in requests], [item['response_token_ids'] for item in originals])

    def test_parallel_array_saved_layout_retains_primary_gold_for_labeling(self):
        payload = {'id': ['x'], 'question': ['Q'], 'gold_answer': ['yes'],
                   'pred': ['Yes, it is.'], 'fixed_hop_trace': [{}]}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'saved.json'
            source.write_text(json.dumps(payload), encoding='utf-8')
            record = read_records(str(source))[0]
        self.assertEqual(enrich_record(record)['hallucination_label'], 0)

    def test_old_label_calibration_is_rejected_before_gpu_loading(self):
        args = _parser().parse_args([
            "evaluate", "--input", "input.jsonl", "--output", "out.jsonl", "--calibration", "fit.json",
        ])
        settings = _runtime_settings(args)
        calibrator = ReDeEPCalibrator()
        calibrator.runtime_settings = {key: value for key, value in settings.items() if key != "label_rule_version"}
        with self.assertRaisesRegex(ValueError, "label_rule_version"):
            calibrator.validate_runtime(settings)
        calibrator.runtime_settings = dict(settings)
        calibrator.validate_runtime(settings)

    def test_metrics_include_threshold_free_auprc(self):
        records = [
            {
                "hallucination_label": label,
                "redeep_score": score,
                "redeep_prediction": prediction,
                "hop_num": 2,
            }
            for label, score, prediction in [
                (0, 0.1, 0),
                (0, 0.4, 1),
                (1, 0.35, 0),
                (1, 0.8, 1),
            ]
        ]
        metrics = _metrics(records)
        self.assertAlmostEqual(metrics["auprc"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["by_hop"]["2"]["auprc"], 5.0 / 6.0)

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
