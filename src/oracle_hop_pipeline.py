"""Sequential full-paragraph generation with cumulative evidence (strategy B)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .data_adapters import load_dataset
from .generator import Generator
from .hallucination_labels import DEFAULT_F1_THRESHOLD, answer_label_fields
from .prompt_utils import fit_prompt_parts


def _evidence_text(hops: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"[step={step} doc_id={doc['doc_id']} title={doc['title']}] {doc['text']}"
        for step, hop in enumerate(hops, start=1) for doc in hop["documents"]
    )


class OracleIterativePipeline:
    """Add one paragraph at a time and retain prior paragraphs and responses."""

    def __init__(self, generator: Generator, f1_threshold: float = DEFAULT_F1_THRESHOLD,
                 max_input_len: Optional[int] = None, allow_evidence_truncation: bool = False):
        self.generator = generator
        self.f1_threshold = float(f1_threshold)
        self.max_input_len = max_input_len or getattr(generator, "max_input_len", None)
        self.allow_evidence_truncation = bool(allow_evidence_truncation)
        if not 0.0 <= self.f1_threshold <= 1.0:
            raise ValueError("f1_threshold must be in [0, 1]")

    def _parts(self, sample, position: int, previous: List[str]) -> Dict[str, Any]:
        prefix = (
            "Solve the question as supporting paragraphs arrive one at a time. "
            "Use only the supplied evidence.\n"
            f"Original question: {sample.question}\n"
            "Previous model responses:\n"
            + ("\n".join(f"Response {i}: {value}" for i, value in enumerate(previous, 1))
               or "(none; this is the first step)")
            + f"\nCurrent step: {position}/{len(sample.hops)}\n"
            "Evidence available so far:\n"
        )
        context = _evidence_text(sample.hops[:position])
        if position == len(sample.hops):
            instruction = "Answer the original question. Return only the shortest answer span, with no explanation."
        else:
            instruction = (
                "Record the concise facts from the available evidence needed to solve the original question. "
                "Preserve relevant facts from previous steps. Do not give the final answer yet."
            )
        suffix = f"\n{instruction}\nResponse:"
        if hasattr(self.generator, "prepare_prompt_parts"):
            parts = self.generator.prepare_prompt_parts(prefix, context, suffix, self.max_input_len)
        else:
            tokenizer = getattr(self.generator, "tokenizer", None)
            parts = fit_prompt_parts(tokenizer, prefix, context, suffix,
                                     self.max_input_len if tokenizer is not None else None)
        if parts.get("context_truncated") and not self.allow_evidence_truncation:
            raise ValueError(
                f"Full evidence exceeds --max-input-len for {sample.sample_id}, step {position}. "
                "Increase the limit or explicitly allow evidence truncation."
            )
        return parts

    def _run_batch(self, samples: List[Any]) -> List[Dict[str, Any]]:
        states = []
        for sample in samples:
            if not sample.hops:
                raise ValueError(f"Sample {sample.sample_id} has no supporting paragraphs")
            states.append({"sample": sample, "previous": [], "records": []})
        if not states:
            return []
        for position in range(1, max(len(s["sample"].hops) for s in states) + 1):
            active = [s for s in states if position <= len(s["sample"].hops)]
            parts_list = [self._parts(s["sample"], position, s["previous"]) for s in active]
            prompts = [parts["prompt"] for parts in parts_list]
            if hasattr(self.generator, "generate_with_token_ids_batch"):
                generated = self.generator.generate_with_token_ids_batch(prompts)
            elif hasattr(self.generator, "generate_with_token_ids"):
                generated = [self.generator.generate_with_token_ids(prompt) for prompt in prompts]
            else:
                generated = [(self.generator.generate(prompt), []) for prompt in prompts]
            if len(generated) != len(active):
                raise RuntimeError("Generator returned an incorrect number of batch responses")
            for state, parts, (response, token_ids) in zip(active, parts_list, generated):
                sample = state["sample"]
                response = str(response).strip()
                visible = sample.hops[:position]
                complete = not parts.get("context_truncated", False)
                state["records"].append({
                    "hop": position, "step": position, "is_final": position == len(sample.hops),
                    "prompt": parts["prompt"], "prompt_parts": parts,
                    "new_document": sample.hops[position - 1]["documents"][0],
                    "evidence": {"hop": position, "documents": [d for h in visible for d in h["documents"]]},
                    "visible_doc_ids": [d["doc_id"] for h in visible for d in h["documents"]],
                    "evidence_complete": complete, "response": response,
                    "response_token_ids": token_ids,
                })
                state["previous"].append(response)
        results = []
        for state in states:
            sample, records = state["sample"], state["records"]
            final = records[-1]
            complete = all(record["evidence_complete"] for record in records)
            results.append({
                "id": sample.sample_id, "dataset": sample.raw.get("dataset", ""),
                "split": sample.raw.get("split", ""), "question": sample.question,
                "gold_answer": sample.answer, "gold_answers": sample.gold_answers,
                # Retained for the existing ReDeEP API; this alias means execution steps.
                "hop_num": len(sample.hops), "num_steps": len(sample.hops),
                "hop_num_definition": "execution_steps",
                "dataset_hop_count": sample.raw.get("dataset_hop_count"),
                "order_source": sample.raw.get("order_source"),
                "prediction": final["response"], "response_token_ids": final["response_token_ids"],
                "answer_prompt": final["prompt"], "prompt_parts": final["prompt_parts"],
                "trace": {"hop_num": len(sample.hops), "mode": "oracle_cumulative",
                          "retriever": None, "hops": sample.hops},
                "hop_records": records, "evidence_mode": "oracle_gold",
                "hop_mode": "iterative", "history_mode": "cumulative_evidence_and_responses",
                "evidence_complete": complete, "retrieval_sufficient": complete,
                "metadata": sample.raw.get("metadata", {}),
                **answer_label_fields(final["response"], sample.gold_answers, self.f1_threshold),
            })
        return results

    def run_sample(self, sample) -> Dict[str, Any]:
        return self._run_batch([sample])[0]

    def run_samples(self, samples: List[Any], batch_size: int = 1, progress_every: int = 10,
                    batch_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None):
        if batch_size < 1 or progress_every < 1:
            raise ValueError("batch_size and progress_every must be positive")
        results = []
        last_report = 0
        for start in range(0, len(samples), batch_size):
            batch = self._run_batch(samples[start:start + batch_size])
            if batch_callback is None:
                results.extend(batch)
            else:
                batch_callback(batch)
            completed = start + len(batch)
            if completed == len(samples) or completed - last_report >= progress_every:
                print(f"[oracle] completed {completed}/{len(samples)} samples "
                      f"({100 * completed / len(samples):.1f}%)", flush=True)
                last_report = completed
        return results


def run_oracle_dataset(input_path: str, dataset_name: str, model_path: str,
                       max_records: Optional[int] = None, f1_threshold: float = DEFAULT_F1_THRESHOLD,
                       max_input_len: int = 3840, max_new_tokens: int = 128,
                       sampling_strategy: str = "prefix", seed: int = 42, batch_size: int = 1,
                       progress_every: int = 10, allow_evidence_truncation: bool = False,
                       batch_callback=None):
    samples = load_dataset(input_path, dataset_name, max_records=max_records,
                           sampling_strategy=sampling_strategy, seed=seed)
    if any(sample.raw.get("schema_version") != 1 for sample in samples):
        raise ValueError("The cumulative-evidence entry point requires data/processed schema_version=1 input")
    generator = Generator(model_path, max_new_tokens=max_new_tokens, max_input_len=max_input_len)
    pipeline = OracleIterativePipeline(generator, f1_threshold, max_input_len, allow_evidence_truncation)
    return pipeline.run_samples(samples, batch_size, progress_every, batch_callback)
