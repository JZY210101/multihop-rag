"""FlashRAG 原生组件上的固定 hop 多跳 Pipeline。"""
import copy
from typing import Any
from flashrag.pipeline import BasicPipeline
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_generator, get_retriever

def _value(obj: Any, keys, default=None):
    for key in keys:
        if isinstance(obj, dict) and key in obj and obj[key] is not None: return obj[key]
        try:
            value = getattr(obj, key)
            if value is not None: return value
        except (AttributeError, KeyError): pass
    return default

def infer_hop_num(item, default=2) -> int:
    raw, meta = _value(item, ["data"], {}) or {}, _value(item, ["metadata"], {}) or {}
    for obj in [raw, meta]:
        value = _value(obj, ["hop_num", "hop", "num_hops", "num_hop"], None)
        if isinstance(value, int) and value > 0: return value
        decomp = _value(obj, ["decomposition", "question_decomposition", "sub_questions", "subquestions"], None)
        if isinstance(decomp, list) and decomp: return len(decomp)
        facts = _value(obj, ["supporting_facts", "supporting facts", "supportingFacts"], None)
        if isinstance(facts, dict): facts = facts.get("title", facts.get("facts", []))
        if isinstance(facts, list) and facts:
            titles = []
            for fact in facts:
                title = fact[0] if isinstance(fact, (list, tuple)) and fact else fact.get("title") if isinstance(fact, dict) else fact
                if title not in titles: titles.append(title)
            if titles: return len(titles)
    return max(1, int(default))

class FixedHopPipeline(BasicPipeline):
    """每个样本严格执行其标注的 hop_num；没有证据充分性判断和提前停止。"""
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        super().__init__(config, prompt_template or PromptTemplate(config))
        self.retriever, self.generator = retriever or get_retriever(config), generator or get_generator(config)
        self.topk, self.default_hop = int(config.get("retrieval_topk", 5)), int(config.get("default_hop_num", 2))
    @staticmethod
    def _docs_text(docs):
        return "\n".join(f"[{i + 1}] {d.get('contents', d.get('text', ''))}" for i, d in enumerate(docs))
    def run(self, dataset, do_eval=True, pred_process_fun=None):
        predictions, traces = [], []
        for item in dataset:
            hop_num, query, hop_trace, evidence = infer_hop_num(item, self.default_hop), item.question, [], []
            for hop in range(1, hop_num + 1):
                docs = copy.deepcopy(self.retriever.batch_search([query])[0][:self.topk])
                hop_trace.append({"hop": hop, "query": query, "documents": docs}); evidence.extend(docs)
                if hop < hop_num:
                    prompt = "Rewrite the original question as a concise search query for the next hop. Do not answer.\nOriginal question: " + item.question + "\nEvidence:\n" + self._docs_text(evidence) + "\nNext query:"
                    query = self.generator.generate([prompt])[0].strip()
            answer_prompt = "Answer using only the retrieved evidence. Give only the final answer.\nQuestion: " + item.question + "\nEvidence:\n" + self._docs_text(evidence) + "\nAnswer:"
            predictions.append(self.generator.generate([answer_prompt])[0].strip()); traces.append({"hop_num": hop_num, "hops": hop_trace})
        dataset.update_output("pred", predictions); dataset.update_output("fixed_hop_trace", traces); dataset.update_output("hop_num", [t["hop_num"] for t in traces])
        return self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
