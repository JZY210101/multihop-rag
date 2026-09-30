import json
import tempfile
import unittest
from pathlib import Path

from src.data_adapters import load_dataset, normalize_record
from src.generator import Generator
from src.oracle_hop_pipeline import OracleIterativePipeline


class DummyGenerator:
    tokenizer = None
    max_input_len = 4096

    def __init__(self):
        self.prompts = []

    def generate_with_token_ids(self, prompt):
        self.prompts.append(prompt)
        return f"intermediate {len(self.prompts)}", [10 + len(self.prompts)]


class ChatTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.assertions = (tokenize, add_generation_prompt)
        return f"<user>{messages[0]['content']}</user><assistant>"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": list(range(len(text)))}


class ChatGenerator(DummyGenerator):
    tokenizer = ChatTokenizer()

    def prepare_prompt_parts(self, prefix, context, suffix, max_tokens=None):
        return Generator.prepare_prompt_parts(self, prefix, context, suffix, max_tokens)


class OraclePipelineTests(unittest.TestCase):
    def test_iterative_pipeline_uses_ordered_gold_hops(self):
        sample = normalize_record(
            {
                "id": "x",
                "question": "Who?",
                "golden_answers": ["final"],
                "metadata": {
                    "question_decomposition": [
                        {
                            "question": "Who first?",
                            "answer": "first",
                            "support_paragraph": {"title": "Page A", "paragraph_text": "Evidence A", "is_supporting": True},
                        },
                        {
                            "question": "Who finally?",
                            "answer": "final",
                            "support_paragraph": {"title": "Page B", "paragraph_text": "Evidence B", "is_supporting": True},
                        },
                    ]
                },
            },
            index=0,
            default_hop=2,
        )
        generator = DummyGenerator()
        result = OracleIterativePipeline(generator).run_sample(sample)
        self.assertEqual(result["evidence_mode"], "oracle_gold")
        self.assertEqual(result["hop_mode"], "iterative")
        self.assertEqual(len(result["hop_records"]), 2)
        self.assertIn("Evidence A", generator.prompts[0])
        self.assertNotIn("Evidence B", generator.prompts[0])
        self.assertIn("Current sub-question: Who first?", generator.prompts[0])
        self.assertIn("Evidence B", generator.prompts[1])
        self.assertIn("Current sub-question: Who finally?", generator.prompts[1])
        self.assertIn("intermediate 1", generator.prompts[1])

    def test_saved_prompt_is_the_exact_chat_formatted_prompt(self):
        sample = normalize_record(
            {
                "id": "chat",
                "question": "Who?",
                "golden_answers": ["final"],
                "metadata": {
                    "question_decomposition": [
                        {
                            "question": "Who finally?",
                            "answer": "final",
                            "support_paragraph": {
                                "title": "Page A",
                                "paragraph_text": "Evidence A",
                                "is_supporting": True,
                            },
                        }
                    ]
                },
            },
            index=0,
            default_hop=1,
        )
        generator = ChatGenerator()
        result = OracleIterativePipeline(generator).run_sample(sample)
        parts = result["prompt_parts"]
        self.assertTrue(parts["chat_template"])
        self.assertEqual(parts["prompt"], generator.prompts[0])
        self.assertEqual(parts["prompt"], parts["prefix"] + parts["context"] + parts["suffix"])
        self.assertIn("<assistant>", parts["suffix"])

    def test_load_dataset_applies_limit_while_reading(self):
        records = [
            {
                "id": str(index),
                "question": "q",
                "answer": "a",
                "metadata": {
                    "question_decomposition": [
                        {
                            "question": "q",
                            "answer": "a",
                            "support_paragraph": {
                                "title": "p",
                                "paragraph_text": "evidence",
                                "is_supporting": True,
                            },
                        }
                    ]
                },
            }
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            samples = load_dataset(str(path), "musique", max_records=1)
        self.assertEqual([sample.sample_id for sample in samples], ["0"])

    def test_random_dataset_sample_is_deterministic_and_not_a_prefix(self):
        records = [
            {
                "id": str(index),
                "question": "q",
                "answer": "a",
                "metadata": {
                    "question_decomposition": [
                        {
                            "question": "q",
                            "answer": "a",
                            "support_paragraph": {
                                "title": "p",
                                "paragraph_text": "evidence",
                                "is_supporting": True,
                            },
                        }
                    ]
                },
            }
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            first = load_dataset(str(path), "musique", max_records=5, sampling_strategy="random", seed=7)
            second = load_dataset(str(path), "musique", max_records=5, sampling_strategy="random", seed=7)
        first_ids = [sample.sample_id for sample in first]
        self.assertEqual(first_ids, [sample.sample_id for sample in second])
        self.assertNotEqual(first_ids, ["0", "1", "2", "3", "4"])


if __name__ == "__main__":
    unittest.main()
