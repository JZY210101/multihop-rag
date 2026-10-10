import unittest

from src.data_adapters import normalize_record
from src.hallucination_labels import (
    LABEL_RULE_VERSION,
    answer_label_fields,
    answer_token_f1,
    best_answer_f1,
    best_answer_em,
    hallucination_label_from_f1,
    normalize_answer,
    retrieval_sufficient,
)
from src.redeep.io import enrich_record, make_label


class SharedLabelTests(unittest.TestCase):
    def test_quotes_and_periods_normalize_once_to_a_stable_answer(self):
        examples = {
            '"Paris".': 'paris', '"Paris."': 'paris', '"Song of the South".': 'song of the south',
            '"The Brothers Lionheart".': 'the brothers lionheart', '"".': '"".',
            '"Portofino", "Zermatt", and "Saving Grandma".':
                '"portofino", "zermatt", and "saving grandma"',
        }
        for raw, expected in examples.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_answer(raw), expected)
                self.assertEqual(normalize_answer(expected), expected)
        fields = answer_label_fields('Song of the South', ['"Song of the South".'])
        self.assertEqual(fields['answer_em'], 1.0)
        self.assertEqual(fields['hallucination_label'], 0)

    def test_malformed_numeric_spans_are_not_partially_rewritten(self):
        for raw in ['1,234.56,789', '1,23,456', '12,34', '1,2345', '1.234,567']:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_answer(raw), raw)
        self.assertEqual(normalize_answer('3,677.'), '3677')
        self.assertEqual(normalize_answer('$1,234.56'), '$1234.56')

    def test_f1_counts_word_multiplicity_and_partial_answers(self):
        self.assertAlmostEqual(answer_token_f1('red red', 'red'), 2.0 / 3.0)
        self.assertAlmostEqual(answer_token_f1('red red blue', 'red blue'), 0.8)
        fields = answer_label_fields('3,677', ['3,677 seated'])
        self.assertEqual(fields['answer_em'], 0.0)
        self.assertAlmostEqual(fields['answer_f1'], 2.0 / 3.0)
        self.assertEqual(fields['hallucination_label'], 0)
        self.assertEqual(fields['hallucination_label_reason'], 'f1_at_or_above_threshold')

    def test_aliases_choose_the_best_em_and_f1_without_inventing_answers(self):
        self.assertEqual(best_answer_em('SGN', [None, '', 'Tan Son Nhat', 'SGN']), 1.0)
        fields = answer_label_fields('University of Paris', ['Random People', 'Paris'])
        self.assertEqual(fields['answer_em'], 0.0)
        self.assertAlmostEqual(fields['answer_f1'], 0.5)
        self.assertEqual(fields['hallucination_label'], 0)

    def test_boolean_type_comes_from_all_nonempty_normalized_golds(self):
        fields = answer_label_fields('Yes, it is.', [None, ' ', 'Yes.', 'YES'])
        self.assertEqual(fields['boolean_answer_status'], 'single_yes')
        self.assertEqual(fields['hallucination_label'], 0)
        for golds in [['Yes Minister'], ['No Country for Old Men'], ['yes', 'affirmative']]:
            with self.subTest(golds=golds):
                self.assertEqual(answer_label_fields('yes', golds)['boolean_answer_status'], 'not_applicable')

    def test_non_ascii_variations_are_not_literal_boolean_words(self):
        for response in ['yeſ', 'yes\u0301', 'nö', 'no中文']:
            with self.subTest(response=response):
                fields = answer_label_fields(response, ['yes'])
                self.assertEqual(fields['boolean_answer_matches'], [])
                self.assertEqual(fields['hallucination_label_reason'], 'boolean_missing')

    def test_stale_labels_are_recomputed_in_all_compatible_modes(self):
        for prediction, stale_label, expected in [('Yes, it is.', 1, 0), ('Yes, no doubt.', 0, 1)]:
            record = {'question': 'Q', 'prediction': prediction, 'gold_answers': ['yes'],
                      'hallucination_label': stale_label, 'trace': {}}
            for mode in ['f1_answer', 'weak_answer', 'f1_retrieval_aware', 'retrieval_aware']:
                with self.subTest(prediction=prediction, mode=mode):
                    self.assertEqual(make_label(record, {}, mode=mode), expected)
                    self.assertEqual(enrich_record(record, label_mode=mode)['hallucination_label'], expected)

    def test_empty_gold_alias_list_can_fall_back_to_a_valid_primary_answer(self):
        record = {'question': 'Q', 'prediction': 'Yes, it is.', 'gold_answers': [None, ' '],
                  'gold_answer': 'yes', 'trace': {}}
        self.assertEqual(normalize_record(record, 0, 2).gold_answers, ['yes'])
        self.assertEqual(enrich_record(record)['gold_answers'], ['yes'])
        self.assertEqual(enrich_record(record)['hallucination_label'], 0)

    def test_all_supported_gold_field_layouts_reach_the_same_label(self):
        for key in ['gold_answers', 'golden_answers', 'gold_answer', 'answer', 'target']:
            for nesting in [None, 'metadata', 'raw']:
                with self.subTest(key=key, nesting=nesting):
                    value = ['yes'] if key.endswith('answers') else 'yes'
                    record = {'question': 'Q', 'prediction': 'YES, it is.', 'trace': {}}
                    if nesting:
                        record[nesting] = {key: value}
                    else:
                        record[key] = value
                    self.assertEqual(enrich_record(record)['hallucination_label'], 0)

    def test_basic_normalization_keeps_articles_and_meaningful_punctuation(self):
        self.assertEqual(normalize_answer("  The United States.  "), "the united states")
        self.assertEqual(normalize_answer("The United States!"), "the united states!")
        self.assertEqual(answer_token_f1("The United States", "United States"), 0.8)
        self.assertEqual(answer_token_f1("wrong entity", "United States"), 0.0)

    def test_real_dataset_answer_atoms_are_preserved(self):
        examples = {
            "A": "a", "The The": "the the", "!!!": "!!!", "$": "$",
            "19.22%": "19.22%", "−9 °F": "-9 °f", "1596-1650": "1596-1650",
            "6.213 km long": "6.213 km long", "$1.7 billion": "$1.7 billion",
            "Searchers 2.0": "searchers 2.0", "Tân Sơn Nhất": "tân sơn nhất",
        }
        for raw, expected in examples.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_answer(raw), expected)
                self.assertEqual(answer_label_fields(raw, [raw])["answer_em"], 1.0)
        self.assertNotEqual(normalize_answer("19.22%"), normalize_answer("1922"))
        self.assertNotEqual(normalize_answer("-9"), normalize_answer("9"))

    def test_thousands_quotes_typography_and_unicode(self):
        examples = {
            "3,677": "3677", "1,234,567.89": "1234567.89", "12,34": "12,34",
            "1,23,456": "1,23,456", "1,2345": "1,2345",
            "“Paris.”": "paris", "‘O’Connor’": "o'connor", '""': '""',
            "Cafe\u0301": "café", "A‑B": "a-b", "October 6": "october 6",
        }
        for raw, expected in examples.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_answer(raw), expected)

    def test_limited_final_period_does_not_strip_initials_or_versions(self):
        for raw in ["U.S.", "A.", "Dr.", "0.9 s.", "3.2.1.", "!!!", "."]:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_answer(raw), raw.lower())
        self.assertEqual(normalize_answer("Paris."), "paris")
        self.assertEqual(normalize_answer("3.14."), "3.14")

    def test_best_f1_uses_all_gold_aliases(self):
        score = best_answer_f1("USA", ["United States of America", "USA"])
        self.assertEqual(score, 1.0)

    def test_threshold_boundary_is_non_hallucination(self):
        prediction = "one two three p1 p2 p3 p4 p5 p6 p7"
        reference = "one two three g1 g2 g3 g4 g5 g6 g7"
        score = answer_token_f1(prediction, reference)
        self.assertAlmostEqual(score, 0.3)
        self.assertEqual(hallucination_label_from_f1(score, 0.3), 0)
        self.assertEqual(answer_label_fields(prediction, [reference])["hallucination_label"], 0)
        self.assertEqual(hallucination_label_from_f1(0.299999, 0.3), 1)
        self.assertEqual(hallucination_label_from_f1(0.300001, 0.3), 0)

    def test_em_and_f1_use_same_normalization_and_all_gold_aliases(self):
        fields = answer_label_fields("TÂN SƠN NHẤT", ["SGN", "Tân Sơn Nhất"])
        self.assertEqual(fields["answer_em"], 1.0)
        self.assertEqual(fields["answer_f1"], 1.0)
        self.assertEqual(fields["hallucination_label"], 0)
        self.assertEqual(fields["hallucination_label_reason"], "exact_match")
        self.assertEqual(fields["hallucination_label_version"], LABEL_RULE_VERSION)

    def test_boolean_extraction_accepts_single_polarity_without_templates(self):
        for raw, expected in [
            ("Yes, it is.", "yes"), ("I think YES.", "yes"), ("yes yes YES", "yes"),
            ("No, it isn't.", "no"), ("The answer is NO!", "no"), ("‘Yes’", "yes"),
            ("Yes, nobody objects.", "yes"), ("No, yesterday was different.", "no"),
        ]:
            with self.subTest(raw=raw):
                fields = answer_label_fields(raw, [expected.upper()])
                self.assertEqual(fields["answer_for_label"], expected)
                self.assertEqual(fields["boolean_answer_matches"], [expected])
                self.assertEqual(fields["answer_em"], 1.0)
                self.assertEqual(fields["answer_f1"], 1.0)
                self.assertEqual(fields["hallucination_label"], 0)

    def test_boolean_opposite_polarities_fail_independent_of_threshold(self):
        for raw in ["Yes, no doubt.", "Yes, there is no difference.", "NO. Actually yes.", "yes/no"]:
            for threshold in [0.0, 0.3, 1.0]:
                with self.subTest(raw=raw, threshold=threshold):
                    fields = answer_label_fields(raw, ["yes"], threshold)
                    self.assertEqual(fields["hallucination_label"], 1)
                    self.assertEqual(fields["hallucination_label_reason"], "boolean_conflict")
                    self.assertEqual(fields["boolean_answer_status"], "both_yes_and_no")
                    self.assertEqual(set(fields["boolean_answer_matches"]), {"yes", "no"})

    def test_boolean_matching_requires_a_whole_word(self):
        for raw in ["yesterday nobody knows", "yesman", "nope", "yes_no", "yes2", "nobody", "yes中文"]:
            with self.subTest(raw=raw):
                fields = answer_label_fields(raw, ["yes"])
                self.assertEqual(fields["boolean_answer_matches"], [])
                self.assertEqual(fields["hallucination_label"], 1)
                self.assertEqual(fields["hallucination_label_reason"], "boolean_missing")

    def test_boolean_wrong_polarity_and_missing_answer_fail_at_zero_threshold(self):
        for raw, reason in [("No, it isn't.", "boolean_mismatch"), ("It is true.", "boolean_missing"),
                            ("", "boolean_missing")]:
            with self.subTest(raw=raw):
                fields = answer_label_fields(raw, ["yes"], threshold=0.0)
                self.assertEqual(fields["hallucination_label"], 1)
                self.assertEqual(fields["hallucination_label_reason"], reason)

    def test_entity_answers_do_not_enable_boolean_extraction(self):
        fields = answer_label_fields("No Country for Old Men", ["No Country for Old Men"])
        self.assertEqual(fields["answer_for_label"], "No Country for Old Men")
        self.assertEqual(fields["boolean_answer_status"], "not_applicable")
        self.assertEqual(fields["hallucination_label"], 0)

    def test_empty_prediction_cannot_pass_even_at_zero_threshold(self):
        fields = answer_label_fields(" \n ", ["Paris"], threshold=0.0)
        self.assertEqual(fields["answer_em"], 0.0)
        self.assertEqual(fields["answer_f1"], 0.0)
        self.assertEqual(fields["hallucination_label"], 1)
        self.assertEqual(answer_token_f1("", ""), 0.0)

    def test_invalid_threshold_is_rejected_even_for_boolean_or_missing_gold(self):
        for threshold in [-0.1, 1.1, float("nan")]:
            for answers in [[], ["yes"], ["Paris"]]:
                with self.subTest(threshold=threshold, answers=answers):
                    with self.assertRaises(ValueError):
                        answer_label_fields("yes", answers, threshold)

    def test_missing_gold_has_no_label(self):
        fields = answer_label_fields("anything", [], 0.3)
        self.assertIsNone(fields["answer_f1"])
        self.assertIsNone(fields["answer_em"])
        self.assertIsNone(fields["hallucination_label"])
        self.assertEqual(fields["hallucination_label_reason"], "missing_gold_answers")

    def test_redeep_preserves_raw_response_and_propagates_shared_label_fields(self):
        for prediction in ["Yes, it is.", "Yes, there is no difference."]:
            with self.subTest(prediction=prediction):
                record = {
                    "question": "Question?", "prediction": prediction,
                    "gold_answers": ["yes"], "response_token_ids": [10, 11, 12], "trace": {},
                }
                enriched = enrich_record(record)
                self.assertEqual(enriched["prediction"], prediction)
                self.assertEqual(enriched["response_token_ids"], [10, 11, 12])
                self.assertEqual(record["prediction"], prediction)
                for key, value in answer_label_fields(prediction, ["yes"]).items():
                    self.assertEqual(enriched[key], value)

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

if __name__ == "__main__":
    unittest.main()
