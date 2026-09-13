from typing import Any, Dict, List
from .hallucination_labels import DEFAULT_F1_THRESHOLD, answer_label_fields, retrieval_sufficient
from .prompt_utils import fit_prompt_parts
from .schema import Sample


class FixedHopPipeline:
    """严格执行每个样本标注的 hop_num，不提前停止。"""

    def __init__(
        self,
        retriever,
        generator,
        top_k: int = 5,
        f1_threshold: float = DEFAULT_F1_THRESHOLD,
    ):
        self.retriever, self.generator, self.top_k = retriever, generator, top_k
        self.f1_threshold = float(f1_threshold)
        if not 0.0 <= self.f1_threshold <= 1.0:
            raise ValueError("f1_threshold must be in [0, 1]")

    @staticmethod
    def _evidence_text(hops: List[Dict[str, Any]]) -> str:
        return "\n".join(
            f"[hop={hop['hop']} doc_id={document['doc_id']}] {document['text']}"
            for hop in hops
            for document in hop["documents"]
        )

    def _prompt_parts(self, prefix: str, context: str, suffix: str) -> Dict[str, Any]:
        tokenizer = getattr(self.generator, "tokenizer", None)
        max_input_len = getattr(self.generator, "max_input_len", None)
        return fit_prompt_parts(tokenizer, prefix, context, suffix, max_input_len if tokenizer is not None else None)

    def run(self, sample: Sample) -> Dict[str, Any]:
        traces, evidence, query = [], [], sample.question
        for hop in range(1, sample.hop_num + 1):
            docs = self.retriever.search(query, self.top_k)
            trace_docs = [{"doc_id": d.doc_id, "text": d.text, "score": d.score, "metadata": d.metadata} for d in docs]
            traces.append({"hop": hop, "query": query, "documents": trace_docs})
            evidence.append({"hop": hop, "documents": trace_docs})
            if hop < sample.hop_num:
                query_parts = self._prompt_parts(
                    f"Rewrite the question into a focused search query for hop {hop + 1}.\n"
                    f"Question: {sample.question}\nEvidence so far:\n",
                    self._evidence_text(evidence),
                    "\nQuery:",
                )
                query = self.generator.generate(query_parts["prompt"]).strip() or sample.question
        prompt_parts = self._prompt_parts(
            "Answer using only the retrieved evidence. Give only the final answer.\n"
            f"Question: {sample.question}\nEvidence:\n",
            self._evidence_text(evidence),
            "\nAnswer:",
        )
        answer_prompt = prompt_parts["prompt"]
        if hasattr(self.generator, "generate_with_token_ids"):
            prediction, response_token_ids = self.generator.generate_with_token_ids(answer_prompt)
        else:
            prediction, response_token_ids = self.generator.generate(answer_prompt), []
        answers = sample.gold_answers or ([sample.answer] if sample.answer else [])
        label_fields = answer_label_fields(prediction, answers, self.f1_threshold)
        result = {
            "id": sample.sample_id,
            "question": sample.question,
            "gold_answer": sample.answer,
            "gold_answers": answers,
            "supporting_facts": sample.supporting_facts,
            "hop_num": sample.hop_num,
            "prediction": prediction,
            "response_token_ids": response_token_ids,
            "trace": traces,
            "answer_prompt": answer_prompt,
            "prompt_parts": prompt_parts,
            **label_fields,
        }
        result["retrieval_sufficient"] = retrieval_sufficient(result, {"hops": traces})
        return result
