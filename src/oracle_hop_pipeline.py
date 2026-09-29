"""Oracle-evidence, iterative multi-hop generation pipeline.

This pipeline deliberately has no retriever.  It consumes the ordered gold
supporting documents extracted by :mod:`src.data_adapters`, generates one
intermediate result per hop, and stores prompt/response traces for ReDeeP.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
            previous = "\n".join(
                f"Intermediate result {i}: {result}" for i, result in enumerate(previous_results, start=1)
            ) or "(none; this is the first hop)"
            is_final = position == total_hops
            instruction = (
                "Produce the final answer to the original question. Give only the answer."
                if is_final
                else "Produce the concise intermediate fact needed by the next hop. Do not give the final answer yet."
            )
            parts = self._parts(
                "Solve the multi-hop question one hop at a time. Use only the supplied gold evidence.\n"
                f"Original question: {sample.question}\n"
                f"Previous intermediate results:\n{previous}\n"
                f"Current hop: {position}/{total_hops}\n",
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


def run_oracle_dataset(
    input_path: str,
    dataset_name: str,
    model_path: str,
    max_records: Optional[int] = None,
    f1_threshold: float = DEFAULT_F1_THRESHOLD,
    max_input_len: int = 3840,
    max_new_tokens: int = 128,
    allow_context_fallback: bool = False,
) -> List[Dict[str, Any]]:
    samples = load_dataset(input_path, dataset_name, allow_context_fallback=allow_context_fallback)
    if max_records is not None:
        samples = samples[: max(0, int(max_records))]
    generator = Generator(model_path, max_new_tokens=max_new_tokens, max_input_len=max_input_len)
    pipeline = OracleIterativePipeline(generator, f1_threshold=f1_threshold, max_input_len=max_input_len)
    return [pipeline.run_sample(sample) for sample in samples]
