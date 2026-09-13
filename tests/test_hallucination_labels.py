import unittest

from src.data_adapters import normalize_record
from src.hallucination_labels import (
    answer_label_fields,
    answer_token_f1,
    best_answer_f1,
    hallucination_label_from_f1,
    normalize_answer,
    retrieval_sufficient,
)
from src.pipeline import FixedHopPipeline
from src.redeep.io import enrich_record
from src.schema import RetrievedDoc, Sample


class SharedLabelTests(unittest.TestCase):
    def test_hotpot_normalization_and_token_f1(self):
        self.assertEqual(normalize_answer("The United States!"), "united states")
        self.assertEqual(answer_token_f1("The United States", "United States"), 1.0)
        self.assertEqual(answer_token_f1("wrong entity", "United States"), 0.0)

    def test_best_f1_uses_all_gold_aliases(self):
        score = best_answer_f1("USA", ["United States of America", "USA"])
        self.assertEqual(score, 1.0)

    def test_threshold_boundary_is_hallucination(self):
        prediction = "one two three p1 p2 p3 p4 p5 p6 p7"
        reference = "one two three g1 g2 g3 g4 g5 g6 g7"
        score = answer_token_f1(prediction, reference)
        self.assertAlmostEqual(score, 0.3)
        self.assertEqual(hallucination_label_from_f1(score, 0.3), 1)
        self.assertEqual(hallucination_label_from_f1(0.300001, 0.3), 0)

    def test_missing_gold_has_no_label(self):
        fields = answer_label_fields("anything", [], 0.3)
        self.assertIsNone(fields["answer_f1"])
        self.assertIsNone(fields["hallucination_label"])

    def test_retrieval_sufficiency_matches_support_titles(self):
        record = {"supporting_facts": [["Required Page", 0]]}
        missing_trace = {"hops": [{"documents": [{"title": "Other Page"}]}]}
        matching_trace = {"hops": [{"documents": [{"metadata": {"title": "Required Page"}}]}]}
        self.assertFalse(retrieval_sufficient(record, missing_trace))
        self.assertTrue(retrieval_sufficient(record, matching_trace))

    def test_redeep_retrieval_mode_gates_shared_f1_label(self):
        record = {
            "question": "Question?",
            "prediction": "correct",
            "gold_answers": ["correct"],
            "supporting_facts": [["Required Page", 0]],
            "trace": {"hops": [{"hop": 1, "documents": [{"title": "Other Page", "text": "x"}]}]},
            "hallucination_label": 0,
        }
        enriched = enrich_record(record, label_mode="f1_retrieval_aware", f1_threshold=0.3)
        self.assertEqual(enriched["answer_f1"], 1.0)
        self.assertEqual(enriched["base_hallucination_label"], 0)
        self.assertIsNone(enriched["hallucination_label"])
        self.assertFalse(enriched["retrieval_sufficient"])

    def test_adapter_preserves_all_gold_answers(self):
        sample = normalize_record(
            {"id": "1", "question": "Q", "golden_answers": ["first", "alias"]},
            index=0,
            default_hop=2,
        )
        self.assertEqual(sample.answer, "first")
        self.assertEqual(sample.gold_answers, ["first", "alias"])

    def test_generic_pipeline_outputs_shared_label_fields(self):
        class Retriever:
            def search(self, query, top_k):
                return [RetrievedDoc("doc", "evidence", 1.0, {"title": "Required Page"})]

        class Generator:
            def generate(self, prompt):
                return "alias"

        sample = Sample(
            sample_id="1",
            question="Q",
            answer="first",
            gold_answers=["first", "alias"],
            hop_num=1,
            supporting_facts=[["Required Page", 0]],
        )
        result = FixedHopPipeline(Retriever(), Generator()).run(sample)
        self.assertEqual(result["answer_f1"], 1.0)
        self.assertEqual(result["hallucination_label"], 0)
        self.assertEqual(result["hallucination_label_method"], "answer_f1_threshold")
        self.assertTrue(result["retrieval_sufficient"])


if __name__ == "__main__":
    unittest.main()
