import json
import tempfile
import unittest
from pathlib import Path

from src.data_adapters import normalize_record
from src.original_datasets import convert_record, read_source
from src.oracle_hop_pipeline import OracleIterativePipeline
from src.redeep.io import enrich_record


def hotpot_record():
    return {
        "_id": "original-id", "type": "bridge", "level": "hard",
        "question": "Tysons Galleria is located in what county?", "answer": "Fairfax County",
        "context": [
            ["McLean, Virginia", ["McLean is in Fairfax County.", "Extra original detail."]],
            ["Distractor", ["UNRELATED_DOCUMENT_SECRET"]],
            ["Tysons Galleria", ["Tysons Galleria is in McLean, Virginia.", "It opened in 1988."]],
        ],
        "supporting_facts": [["McLean, Virginia", 0], ["Tysons Galleria", 0]],
    }


class DummyGenerator:
    tokenizer = None
    max_input_len = 4096

    def __init__(self):
        self.prompts = []

    def generate_with_token_ids_batch(self, prompts):
        self.prompts.extend(prompts)
        return [(f"MODEL_RESPONSE_{len(self.prompts)}", [123]) for _ in prompts]


class OriginalDatasetTests(unittest.TestCase):
    def test_generation_and_redeep_share_boolean_labels_without_replacing_responses(self):
        raw = hotpot_record()
        raw['answer'] = 'yes'
        record = convert_record(raw, 'hotpotqa', 'dev', 'source.json', 0)
        sample = normalize_record(record, 0, 2)

        class BooleanGenerator(DummyGenerator):
            def __init__(self, response):
                super().__init__()
                self.response = response

            def generate_with_token_ids_batch(self, prompts):
                self.prompts.extend(prompts)
                return [(self.response, [10, 11, 12]) for _ in prompts]

        for response, expected in [('Yes, it is.', 0), ('Yes, there is no difference.', 1)]:
            with self.subTest(response=response):
                output = OracleIterativePipeline(BooleanGenerator(response)).run_sample(sample)
                enriched = enrich_record(output)
                self.assertEqual(output['prediction'], response)
                self.assertEqual(output['hop_records'][-1]['response'], response)
                self.assertEqual(enriched['prediction'], response)
                self.assertEqual(enriched['response_token_ids'], [10, 11, 12])
                for field in ('answer_for_label', 'answer_em', 'answer_f1', 'hallucination_label',
                              'hallucination_label_reason', 'hallucination_label_version'):
                    self.assertEqual(enriched[field], output[field])
                self.assertEqual(enriched['hallucination_label'], expected)

    def test_preserves_full_paragraphs_and_original_id_with_no_distractors(self):
        raw = hotpot_record()
        record = convert_record(raw, "hotpotqa", "dev", "source.json", 95)
        self.assertEqual(record["id"], raw["_id"])
        self.assertEqual(record["gold_answers"], [raw["answer"]])
        self.assertEqual([d["title"] for d in record["support_documents"]],
                         ["McLean, Virginia", "Tysons Galleria"])
        self.assertIn("1988", record["support_documents"][1]["text"])
        self.assertNotIn("UNRELATED", json.dumps(record))
        self.assertEqual(record["num_steps"], 2)
        self.assertIsNone(record["dataset_hop_count"])
        self.assertEqual(record["order_source"], "flashrag_supporting_facts_order")

    def test_invalid_sentence_annotations_are_reported_without_pruning(self):
        raw = hotpot_record()
        raw["supporting_facts"][0][1] = 902
        record = convert_record(raw, "hotpotqa", "dev", "source.json", 0)
        self.assertEqual(record["metadata"]["annotation_warnings"]["invalid_support_sentence_references"],
                         [["McLean, Virginia", 902]])
        self.assertIn("Extra original detail", record["support_documents"][0]["text"])

    def test_2wiki_keeps_annotation_order_instead_of_sorting_by_relations(self):
        raw = {
            "_id": "wiki-id", "type": "compositional", "question": "Who is Film A's director's mother?",
            "answer": "GOLD_ANSWER_SECRET",
            "context": [["Director B", ["Director B's mother is C."]],
                        ["Film A (film)", ["Film A was directed by Director B."]]],
            "supporting_facts": [["Director B", 0], ["Film A (film)", 0]],
            "evidences": [["Film A", "director", "Director B"],
                          ["Director B", "mother", "C"], ["Director B", "mother", "Alias C"]],
        }
        record = convert_record(raw, "2wikimultihopqa", "train", "source.json", 0)
        self.assertEqual(record["support_documents"][0]["title"], "Director B")
        self.assertEqual(record["order_source"], "flashrag_supporting_facts_order")
        self.assertEqual(record["num_steps"], 2)
        self.assertIsNone(record["dataset_hop_count"])
        self.assertEqual(len(record["metadata"]["evidences"]), 3)

    def test_hotpot_annotation_order_does_not_require_a_question_title_match(self):
        raw = hotpot_record()
        raw["question"] = "Where is the place?"
        raw["context"][2][1] = ["The location is unspecified."]
        record = convert_record(raw, "hotpotqa", "dev", "source.json", 0)
        self.assertEqual(record["order_source"], "flashrag_supporting_facts_order")
        self.assertFalse(record["metadata"]["order_details"]["needs_order_review"])
        self.assertFalse(record["metadata"]["order_details"]["is_official_reasoning_order"])
        self.assertEqual([d["title"] for d in record["support_documents"]],
                         list(dict.fromkeys(title for title, _ in raw["supporting_facts"])))

    def test_2wiki_cyclic_relations_preserve_sample_and_support_order(self):
        raw = {
            "_id": "7a2249120bb011ebab90acde48001122", "type": "compositional", "question": "Who directed Indian (2001 film)?",
            "answer": "N. Maharajan", "context": [["Indian (2001 film)", ["Directed by N. Maharajan."]],
                                                       ["N. Maharajan", ["He is Indian."]]],
            "supporting_facts": [["Indian (2001 film)", 0], ["N. Maharajan", 0]],
            "evidences": [["Indian (2001 film)", "director", "N. Maharajan"],
                          ["N. Maharajan", "father", "Indian (2001 film)"]],
        }
        record = convert_record(raw, "2wikimultihopqa", "dev", "source.json", 0)
        self.assertEqual(record["id"], raw["_id"])
        self.assertEqual([d["title"] for d in record["support_documents"]],
                         ["Indian (2001 film)", "N. Maharajan"])
        self.assertEqual(record["metadata"]["evidences"], raw["evidences"])
        self.assertFalse(record["metadata"]["order_details"]["needs_order_review"])

    def test_musique_uses_support_indices_and_original_aliases(self):
        raw = {
            "id": "2hop__a_b", "question": "Who?", "answer": "Final", "answer_aliases": ["Other name"],
            "answerable": True,
            "paragraphs": [{"idx": 0, "title": "B", "paragraph_text": "Evidence B", "is_supporting": True},
                           {"idx": 1, "title": "Noise", "paragraph_text": "Noise", "is_supporting": False},
                           {"idx": 2, "title": "A", "paragraph_text": "Evidence A", "is_supporting": True}],
            "question_decomposition": [{"id": 1, "question": "first", "answer": "SECRET_INTERMEDIATE",
                                        "paragraph_support_idx": 2},
                                       {"id": 2, "question": "#1 next", "answer": "Final",
                                        "paragraph_support_idx": 0}],
        }
        record = convert_record(raw, "musique", "dev", "source.jsonl", 0)
        self.assertEqual([d["source_index"] for d in record["support_documents"]], [2, 0])
        self.assertEqual(record["dataset_hop_count"], 2)
        self.assertEqual(record["gold_answers"], ["Final", "Other name"])
        sample = normalize_record(record, 0, 2)
        generator = DummyGenerator()
        OracleIterativePipeline(generator).run_sample(sample)
        self.assertNotIn("SECRET_INTERMEDIATE", " ".join(generator.prompts))
        raw["answerable"] = False
        with self.assertRaisesRegex(ValueError, "answerable"):
            convert_record(raw, "musique", "dev", "source.jsonl", 0)

    def test_strategy_b_has_history_full_context_and_no_future_evidence(self):
        record = convert_record(hotpot_record(), "hotpotqa", "dev", "source.json", 0)
        sample = normalize_record(record, 0, 2)
        generator = DummyGenerator()
        output = OracleIterativePipeline(generator).run_sample(sample)
        first, final = output["hop_records"]
        self.assertNotIn("1988", first["prompt"])
        self.assertIn("Extra original detail", first["prompt_parts"]["context"])
        self.assertIn("1988", final["prompt_parts"]["context"])
        self.assertIn("Extra original detail", final["prompt_parts"]["context"])
        self.assertIn("MODEL_RESPONSE_1", final["prompt_parts"]["prefix"])
        self.assertNotIn("MODEL_RESPONSE_1", final["prompt_parts"]["context"])
        self.assertEqual(final["visible_doc_ids"], [d["doc_id"] for d in record["support_documents"]])
        self.assertTrue(output["evidence_complete"])
        enriched = enrich_record(output)
        self.assertEqual(enriched["prompt_parts"], final["prompt_parts"])
        self.assertEqual(enriched["history_mode"], "cumulative_evidence_and_responses")
        self.assertEqual(enriched["num_steps"], 2)

    def test_complete_batches_flush_before_next_batch_and_keep_sample_histories_separate(self):
        record = convert_record(hotpot_record(), "hotpotqa", "dev", "source.json", 0)
        samples = [normalize_record({**record, "id": str(i)}, i, 2) for i in range(3)]
        generator = DummyGenerator()
        batches = []
        result = OracleIterativePipeline(generator).run_samples(
            samples, batch_size=2, progress_every=2,
            batch_callback=lambda rows: batches.append([r["id"] for r in rows]))
        self.assertEqual(result, [])
        self.assertEqual(batches, [["0", "1"], ["2"]])
        self.assertIn("1988", generator.prompts[2])
        self.assertNotIn("1988", generator.prompts[4])
        self.assertIn("(none; this is the first step)", generator.prompts[4])

    def test_truncated_evidence_is_rejected_by_default(self):
        class TruncatingGenerator(DummyGenerator):
            def prepare_prompt_parts(self, prefix, context, suffix, limit):
                return {"prompt": prefix + context + suffix, "prefix": prefix,
                        "context": context, "suffix": suffix, "context_truncated": True}
        record = convert_record(hotpot_record(), "hotpotqa", "dev", "source.json", 0)
        sample = normalize_record(record, 0, 2)
        with self.assertRaisesRegex(ValueError, "Full evidence exceeds"):
            OracleIterativePipeline(TruncatingGenerator()).run_sample(sample)
        output = OracleIterativePipeline(TruncatingGenerator(), allow_evidence_truncation=True).run_sample(sample)
        self.assertFalse(output["evidence_complete"])
        self.assertFalse(output["retrieval_sufficient"])

    def test_processed_count_and_missing_paragraphs_are_rejected(self):
        record = convert_record(hotpot_record(), "hotpotqa", "dev", "source.json", 0)
        record["num_steps"] = 3
        with self.assertRaisesRegex(ValueError, "num_steps"):
            normalize_record(record, 0, 2)
        raw = hotpot_record()
        raw["supporting_facts"].append(["Missing paragraph", 0])
        with self.assertRaisesRegex(ValueError, "Missing or ambiguous"):
            convert_record(raw, "hotpotqa", "dev", "source.json", 0)

    def test_source_reader_handles_array_chunk_boundaries(self):
        rows = [{"question": "Unicode 中文", "long": "a" * (1024 * 1024 + 50)}, {"question": "last"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(list(read_source(path)), rows)


if __name__ == "__main__":
    unittest.main()
