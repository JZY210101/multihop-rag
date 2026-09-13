"""FlashRAG 原生组件上的固定 hop 多跳 Pipeline。"""
import copy
from typing import Any
from flashrag.pipeline import BasicPipeline
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_generator, get_retriever

from .hallucination_labels import DEFAULT_F1_THRESHOLD, answer_label_fields, retrieval_sufficient
from .prompt_utils import fit_prompt_parts, normalize_generated_token_ids


def _value(obj: Any, keys, default=None):
    for key in keys:
        if isinstance(obj, dict) and key in obj and obj[key] is not None:
            return obj[key]
        try:
            value = getattr(obj, key)
            if value is not None:
                return value
        except (AttributeError, KeyError):
            pass
    return default


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    try:
        value = config[key]
    except (KeyError, TypeError):
        value = None
    return default if value is None else value


def _make_generator(config):
    # FlashRAG's generic factory currently opens ``generator_model_path`` as a
    # local directory before dispatching. Instantiate the known Qwen HF backend
    # directly so a Hugging Face model id remains valid.
    if _config_value(config, "framework") == "hf":
        from flashrag.generator import HFCausalLMGenerator

        return HFCausalLMGenerator(config)
    return get_generator(config)


def infer_hop_num(item, default=2) -> int:
    raw, meta = _value(item, ["data"], {}) or {}, _value(item, ["metadata"], {}) or {}
    for obj in [raw, meta]:
        value = _value(obj, ["hop_num", "hop", "num_hops", "num_hop"], None)
        if isinstance(value, int) and value > 0:
            return value
        decomp = _value(obj, ["decomposition", "question_decomposition", "sub_questions", "subquestions"], None)
        if isinstance(decomp, list) and decomp:
            return len(decomp)
        facts = _value(obj, ["supporting_facts", "supporting facts", "supportingFacts"], None)
        if isinstance(facts, dict):
            facts = facts.get("title", facts.get("facts", []))
        if isinstance(facts, list) and facts:
            titles = []
            for fact in facts:
                title = (
                    fact[0]
                    if isinstance(fact, (list, tuple)) and fact
                    else fact.get("title")
                    if isinstance(fact, dict)
                    else fact
                )
                if title not in titles:
                    titles.append(title)
            if titles:
                return len(titles)
    return max(1, int(default))


def _gold_answers(item):
    answers = _value(item, ["golden_answers", "gold_answers", "answer", "gold_answer"], None)
    if answers is None:
        raw = _value(item, ["data", "raw"], {}) or {}
        answers = _value(raw, ["golden_answers", "gold_answers", "answer", "gold_answer"], [])
    if not isinstance(answers, list):
        answers = [answers]
    return [str(answer) for answer in answers if answer is not None and str(answer).strip()]


class FixedHopPipeline(BasicPipeline):
    """严格执行标注 hop_num；完成后记录检索充分性，但不据此提前停止。"""

    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template or PromptTemplate(config))
        self.retriever = retriever or get_retriever(config)
        self.generator = generator or _make_generator(config)
        self.topk = int(_config_value(config, "retrieval_topk", 5))
        self.default_hop = int(_config_value(config, "default_hop_num", 2))
        self.f1_threshold = float(_config_value(config, "hallucination_f1_threshold", DEFAULT_F1_THRESHOLD))
        self.generator_max_input_len = int(_config_value(config, "generator_max_input_len", 3840))
        self.prompt_tokenizer = getattr(self.generator, "tokenizer", None)
        if not 0.0 <= self.f1_threshold <= 1.0:
            raise ValueError("hallucination_f1_threshold must be in [0, 1]")
        if self.topk < 1 or self.default_hop < 1:
            raise ValueError("retrieval_topk and default_hop_num must be positive")

    @staticmethod
    def _docs_text(docs):
        """Render evidence while retaining hop/document provenance."""
        lines = []
        for i, item in enumerate(docs):
            if isinstance(item, dict) and "document" in item:
                hop, doc = item.get("hop", "?"), item["document"]
            else:
                hop, doc = "?", item
            if not isinstance(doc, dict):
                doc = {"text": str(doc)}
            doc_id = doc.get("id", doc.get("doc_id", doc.get("title", i + 1)))
            text = doc.get("contents", doc.get("text", doc.get("content", "")))
            lines.append(f"[hop={hop} doc_id={doc_id}] {text}")
        return "\n".join(lines)

    def _generate_answer(self, prompt):
        """Return decoded text and exact generated ids when supported."""
        try:
            generated = self.generator.generate([prompt], return_dict=True)
        except TypeError:
            return self.generator.generate([prompt])[0].strip(), []
        if not isinstance(generated, dict) or "responses" not in generated:
            response = generated[0] if isinstance(generated, list) else generated
            return str(response).strip(), []
        response = str(generated["responses"][0]).strip()
        tokenizer = getattr(self.generator, "tokenizer", None)
        token_ids = normalize_generated_token_ids(
            generated.get("generated_token_ids"),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        )
        return response, token_ids

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        predictions, traces, answer_prompts, redeep_records = [], [], [], []
        answer_f1_values, labels, label_methods, label_thresholds, retrieval_flags = [], [], [], [], []
        for item in dataset:
            hop_num, query, hop_trace = infer_hop_num(item, self.default_hop), item.question, []
            for hop in range(1, hop_num + 1):
                docs = copy.deepcopy(self.retriever.batch_search([query])[0][: self.topk])
                hop_trace.append({"hop": hop, "query": query, "documents": docs})
                evidence_with_provenance = [
                    {"hop": hop_record["hop"], "document": document}
                    for hop_record in hop_trace
                    for document in hop_record["documents"]
                ]
                if hop < hop_num:
                    query_parts = fit_prompt_parts(
                        self.prompt_tokenizer,
                        "Rewrite the original question as a concise search query for the next hop. "
                        f"Do not answer.\nOriginal question: {item.question}\nEvidence:\n",
                        self._docs_text(evidence_with_provenance),
                        "\nNext query:",
                        self.generator_max_input_len if self.prompt_tokenizer is not None else None,
                    )
                    query = self.generator.generate([query_parts["prompt"]])[0].strip()
                    if not query:
                        query = item.question
            prompt_parts = fit_prompt_parts(
                self.prompt_tokenizer,
                "Answer using only the retrieved evidence. Give only the final answer.\n"
                f"Question: {item.question}\nEvidence:\n",
                self._docs_text(evidence_with_provenance),
                "\nAnswer:",
                self.generator_max_input_len if self.prompt_tokenizer is not None else None,
            )
            answer_prompt = prompt_parts["prompt"]
            prediction, response_token_ids = self._generate_answer(answer_prompt)
            trace = {"hop_num": hop_num, "hops": hop_trace}
            answers = _gold_answers(item)
            label_fields = answer_label_fields(prediction, answers, self.f1_threshold)
            predictions.append(prediction)
            traces.append(trace)
            answer_prompts.append(answer_prompt)
            redeep_record = {
                "id": str(getattr(item, "id", getattr(item, "sample_id", len(redeep_records)))),
                "question": item.question,
                "gold_answer": answers[0] if answers else "",
                "gold_answers": copy.deepcopy(answers),
                "metadata": copy.deepcopy(_value(item, ["metadata"], {}) or {}),
                "raw": copy.deepcopy(_value(item, ["data", "raw"], {}) or {}),
                "hop_num": hop_num,
                "trace": trace,
                "answer_prompt": answer_prompt,
                "prompt_parts": prompt_parts,
                "prediction": prediction,
                "response_token_ids": response_token_ids,
                **label_fields,
            }
            sufficient = retrieval_sufficient(redeep_record, trace)
            redeep_record["retrieval_sufficient"] = sufficient
            redeep_records.append(redeep_record)
            answer_f1_values.append(label_fields["answer_f1"])
            labels.append(label_fields["hallucination_label"])
            label_methods.append(label_fields["hallucination_label_method"])
            label_thresholds.append(label_fields["hallucination_f1_threshold"])
            retrieval_flags.append(sufficient)
        dataset.update_output("pred", predictions)
        dataset.update_output("fixed_hop_trace", traces)
        dataset.update_output("answer_prompt", answer_prompts)
        dataset.update_output("redeep_records", redeep_records)
        dataset.update_output("hop_num", [t["hop_num"] for t in traces])
        dataset.update_output("answer_f1", answer_f1_values)
        dataset.update_output("hallucination_label", labels)
        dataset.update_output("hallucination_label_method", label_methods)
        dataset.update_output("hallucination_f1_threshold", label_thresholds)
        dataset.update_output("retrieval_sufficient", retrieval_flags)
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
