import unittest

from src.data_adapters import normalize_record
from src.oracle_hop_pipeline import OracleIterativePipeline


class DummyGenerator:
    tokenizer = None
    max_input_len = 4096

    def __init__(self):
        self.prompts = []

    def generate_with_token_ids(self, prompt):
        self.prompts.append(prompt)
        return f"intermediate {len(self.prompts)}", [10 + len(self.prompts)]


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
        self.assertIn("Evidence B", generator.prompts[1])
        self.assertIn("intermediate 1", generator.prompts[1])


if __name__ == "__main__":
    unittest.main()
