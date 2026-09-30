"""Oracle-evidence, iterative multi-hop generation pipeline.

This pipeline deliberately has no retriever.  It consumes the ordered gold
supporting documents extracted by :mod:`src.data_adapters`, generates one
intermediate result per hop, and stores prompt/response traces for ReDeeP.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .data_adapters import load_dataset
from .hallucination_labels import DEFAULT_F1_THRESHOLD, answer_label_fields
from .generator import Generator
from .prompt_utils import fit_prompt_parts


def _evidence_text(documents: List[Dict[str, Any]], hop: int) -> str:
    lines = []
    for index, document in enumerate(documents, start=1):
        title = document.get("title", document.get("doc_id", index))
        text = document.get("text", document.get("contents", document.get("content", "")))
        lines.append(f"[hop={hop} doc_id={title}] {text}")
    return "\n".join(lines)


class OracleIterativePipeline:
    """Generate with one gold evidence group at a time; retrieval is disabled."""

    def __init__(
        self,
        generator: Generator,
        f1_threshold: float = DEFAULT_F1_THRESHOLD,
        max_input_len: Optional[int] = None,
    ):
        self.generator = generator
        self.f1_threshold = float(f1_threshold)
        self.max_input_len = max_input_len or getattr(generator, "max_input_len", None)
        if not 0.0 <= self.f1_threshold <= 1.0:
            raise ValueError("f1_threshold must be in [0, 1]")

    def _parts(self, prefix: str, context: str, suffix: str) -> Dict[str, Any]:
        if hasattr(self.generator, "prepare_prompt_parts"):
            return self.generator.prepare_prompt_parts(prefix, context, suffix, self.max_input_len)
        return fit_prompt_parts(
            getattr(self.generator, "tokenizer", None),
            prefix,
            context,
            suffix,
            self.max_input_len if getattr(self.generator, "tokenizer", None) is not None else None,
        )

    def _generate(self, prompt: str):
        if hasattr(self.generator, "generate_with_token_ids"):
            return self.generator.generate_with_token_ids(prompt)
        return self.generator.generate(prompt), []

    def run_sample(self, sample) -> Dict[str, Any]:
        if not sample.hops:
            raise ValueError(f"Sample {sample.sample_id} has no gold supporting hops")

        hop_records: List[Dict[str, Any]] = []
        previous_results: List[str] = []
        total_hops = len(sample.hops)
        for position, hop in enumerate(sample.hops, start=1):
            evidence = _evidence_text(hop.get("documents", []), position)
            sub_question = str(hop.get("sub_question", "")).strip()
            previous = "\n".join(
                f"Intermediate result {i}: {result}" for i, result in enumerate(previous_results, start=1)
            ) or "(none; this is the first hop)"
            is_final = position == total_hops
            if is_final:
                instruction = "Answer the original question. Return only the shortest answer span, with no explanation."
            elif sub_question:
                instruction = (
                    "Answer the current sub-question. Return only the concise intermediate answer needed by the next hop."
                )
            else:
                instruction = (
                    "Produce the concise intermediate fact needed by the next hop. Do not answer the original question yet."
                )
            sub_question_line = f"Current sub-question: {sub_question}\n" if sub_question else ""
            parts = self._parts(
                "Solve the multi-hop question one hop at a time. Use only the supplied gold evidence.\n"
                f"Original question: {sample.question}\n"
                f"Previous intermediate results:\n{previous}\n"
                f"Current hop: {position}/{total_hops}\n"
                f"{sub_question_line}",
                evidence,
                f"\n{instruction}\nResponse:",
            )
            response, token_ids = self._generate(parts["prompt"])
            response = str(response).strip()
            hop_records.append({
                "hop": position,
                "is_final": is_final,
                "prompt": parts["prompt"],
                "prompt_parts": parts,
                "evidence": hop,
                "response": response,
                "response_token_ids": token_ids,
                "gold_intermediate_answer": hop.get("gold_intermediate_answer", ""),
            })
            previous_results.append(response)

        final = hop_records[-1]
        labels = answer_label_fields(final["response"], sample.gold_answers, self.f1_threshold)
        trace = {
            "hop_num": total_hops,
            "mode": "oracle_iterative",
            "retriever": None,
            "hops": [
                {
                    "hop": item.get("hop"),
                    "documents": item.get("documents", []),
                }
                for item in sample.hops
            ],
        }
        return {
            "id": sample.sample_id,
            "dataset": sample.raw.get("dataset", "") if isinstance(sample.raw, dict) else "",
            "question": sample.question,
            "gold_answer": sample.answer,
            "gold_answers": sample.gold_answers,
            "hop_num": total_hops,
            "prediction": final["response"],
            "response_token_ids": final["response_token_ids"],
            "answer_prompt": final["prompt"],
            "prompt_parts": final["prompt_parts"],
            "trace": trace,
            "hop_records": hop_records,
            "evidence_mode": "oracle_gold",
            "hop_mode": "iterative",
            "retrieval_sufficient": True,
            **labels,
        }

    def _run_batch(self, samples: List[Any]) -> List[Dict[str, Any]]:
        """Finish every hop for one sample batch before returning it."""
        states = []
        for sample in samples:
            if not sample.hops:
                raise ValueError(f"Sample {sample.sample_id} has no gold supporting hops")
            states.append({"sample": sample, "previous": [], "records": []})

        max_hops = max(len(state["sample"].hops) for state in states)
        for position in range(1, max_hops + 1):
            active = [state for state in states if position <= len(state["sample"].hops)]
            prompts, parts_list = [], []
            for state in active:
                sample = state["sample"]
                hop = sample.hops[position - 1]
                evidence = _evidence_text(hop.get("documents", []), position)
                previous = "\n".join(
                    f"Intermediate result {i}: {result}"
                    for i, result in enumerate(state["previous"], start=1)
                ) or "(none; this is the first hop)"
                is_final = position == len(sample.hops)
                sub_question = str(hop.get("sub_question", "")).strip()
                if is_final:
                    instruction = "Answer the original question. Return only the shortest answer span, with no explanation."
                elif sub_question:
                    instruction = "Answer the current sub-question. Return only the concise intermediate answer needed by the next hop."
                else:
                    instruction = "Produce the concise intermediate fact needed by the next hop. Do not answer the original question yet."
                sub_question_line = f"Current sub-question: {sub_question}\n" if sub_question else ""
                parts = self._parts(
                    "Solve the multi-hop question one hop at a time. Use only the supplied gold evidence.\n"
                    f"Original question: {sample.question}\nPrevious intermediate results:\n{previous}\n"
                    f"Current hop: {position}/{len(sample.hops)}\n{sub_question_line}",
                    evidence,
                    f"\n{instruction}\nResponse:",
                )
                prompts.append(parts["prompt"])
                parts_list.append((state, hop, parts, is_final))

            if hasattr(self.generator, "generate_with_token_ids_batch"):
                generated = self.generator.generate_with_token_ids_batch(prompts)
            else:
                generated = [self._generate(prompt) for prompt in prompts]
            for (state, hop, parts, is_final), (response, token_ids) in zip(parts_list, generated):
                response = str(response).strip()
                state["records"].append({
                    "hop": position,
                    "is_final": is_final,
                    "prompt": parts["prompt"],
                    "prompt_parts": parts,
                    "evidence": hop,
                    "response": response,
                    "response_token_ids": token_ids,
                    "gold_intermediate_answer": hop.get("gold_intermediate_answer", ""),
                })
                state["previous"].append(response)

        batch_results = []
        for state in states:
            sample, records = state["sample"], state["records"]
            final = records[-1]
            labels = answer_label_fields(final["response"], sample.gold_answers, self.f1_threshold)
            batch_results.append({
                "id": sample.sample_id,
                "dataset": sample.raw.get("dataset", "") if isinstance(sample.raw, dict) else "",
                "question": sample.question,
                "gold_answer": sample.answer,
                "gold_answers": sample.gold_answers,
                "hop_num": len(sample.hops),
                "prediction": final["response"],
                "response_token_ids": final["response_token_ids"],
                "answer_prompt": final["prompt"],
                "prompt_parts": final["prompt_parts"],
                "trace": {
                    "hop_num": len(sample.hops),
                    "mode": "oracle_iterative",
                    "retriever": None,
                    "hops": [
                        {"hop": item.get("hop"), "documents": item.get("documents", [])}
                        for item in sample.hops
                    ],
                },
                "hop_records": records,
                "evidence_mode": "oracle_gold",
                "hop_mode": "iterative",
                "retrieval_sufficient": True,
                **labels,
            })
        return batch_results

    def run_samples(
        self,
        samples: List[Any],
        batch_size: int = 1,
        progress_every: int = 10,
        batch_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Complete and optionally persist one full multi-hop batch at a time."""
        samples = list(samples)
        total = len(samples)
        batch_size = max(1, int(batch_size))
        progress_every = max(1, int(progress_every))
        results: List[Dict[str, Any]] = []
        completed = 0
        last_report = 0

        for start in range(0, total, batch_size):
            batch_results = self._run_batch(samples[start : start + batch_size])
            if batch_callback is None:
                results.extend(batch_results)
            else:
                batch_callback(batch_results)
            completed += len(batch_results)
            if completed == total or completed - last_report >= progress_every:
                print(
                    f"[oracle] completed {completed}/{total} samples "
                    f"({100.0 * completed / max(1, total):.1f}%)",
                    flush=True,
                )
                last_report = completed
        return results


def run_oracle_dataset(
    input_path: str,
    dataset_name: str,
    model_path: str,
    max_records: Optional[int] = None,
    f1_threshold: float = DEFAULT_F1_THRESHOLD,
    max_input_len: int = 3840,
    max_new_tokens: int = 128,
    allow_context_fallback: bool = False,
    sampling_strategy: str = "prefix",
    seed: int = 42,
    batch_size: int = 1,
    progress_every: int = 10,
    batch_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
) -> List[Dict[str, Any]]:
    samples = load_dataset(
        input_path,
        dataset_name,
        allow_context_fallback=allow_context_fallback,
        max_records=max_records,
        sampling_strategy=sampling_strategy,
        seed=seed,
    )
    generator = Generator(model_path, max_new_tokens=max_new_tokens, max_input_len=max_input_len)
    pipeline = OracleIterativePipeline(generator, f1_threshold=f1_threshold, max_input_len=max_input_len)
    return pipeline.run_samples(
        samples,
        batch_size=max(1, int(batch_size)),
        progress_every=max(1, int(progress_every)),
        batch_callback=batch_callback,
    )
