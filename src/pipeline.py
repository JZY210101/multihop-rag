from typing import Any, Dict, List
from .schema import Sample

class FixedHopPipeline:
    """严格执行每个样本标注的 hop_num，不提前停止。"""
    def __init__(self, retriever, generator, top_k: int = 5): self.retriever, self.generator, self.top_k = retriever, generator, top_k
    @staticmethod
    def _evidence_text(hops: List[Dict[str, Any]]) -> str:
        return "\n".join(f"[hop {h['hop']}] {d['text']}" for h in hops for d in h["documents"])
    def run(self, sample: Sample) -> Dict[str, Any]:
        traces, evidence, query = [], [], sample.question
        for hop in range(1, sample.hop_num + 1):
            docs = self.retriever.search(query, self.top_k)
            trace_docs = [{"doc_id": d.doc_id, "text": d.text, "score": d.score, "metadata": d.metadata} for d in docs]
            traces.append({"hop": hop, "query": query, "documents": trace_docs}); evidence.append({"hop": hop, "documents": trace_docs})
            if hop < sample.hop_num:
                query = self.generator.generate(f"Rewrite the question into a focused search query for hop {hop + 1}.\nQuestion: {sample.question}\nEvidence so far:\n{self._evidence_text(evidence)}\nQuery:")
        prediction = self.generator.generate(f"Answer the question using only the evidence.\nQuestion: {sample.question}\nEvidence:\n{self._evidence_text(evidence)}\nAnswer:")
        return {"id": sample.sample_id, "question": sample.question, "gold_answer": sample.answer, "hop_num": sample.hop_num, "prediction": prediction, "trace": traces}
