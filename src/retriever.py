import json
from pathlib import Path
from typing import List

from .schema import RetrievedDoc


class BM25Retriever:
    def __init__(self, corpus_path: str):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            BM25Okapi = None
        p = Path(corpus_path)
        if not p.is_file():
            raise FileNotFoundError(f"Corpus file not found: {p}")
        if p.suffix == ".jsonl":
            records = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            obj = json.loads(p.read_text(encoding="utf-8"))
            records = obj if isinstance(obj, list) else obj.get("data", obj.get("documents", []))
        if not records:
            raise ValueError(f"Corpus is empty or has an unsupported layout: {p}")
        self.docs = []
        for index, record in enumerate(records):
            if isinstance(record, str):
                self.docs.append((str(index), record, {}))
            else:
                self.docs.append(
                    (
                        str(record.get("id", record.get("title", index))),
                        record.get("text", record.get("contents", record.get("content", ""))),
                        record,
                    )
                )
        self._tokenized = [d[1].lower().split() for d in self.docs]
        self.bm25 = BM25Okapi(self._tokenized) if BM25Okapi else None

    def search(self, query: str, top_k: int = 5) -> List[RetrievedDoc]:
        if self.bm25:
            scores = self.bm25.get_scores(query.lower().split())
        else:
            # Dependency-free fallback for smoke tests; install rank-bm25 for real runs.
            terms = set(query.lower().split())
            scores = [sum(tok in terms for tok in doc) for doc in self._tokenized]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [RetrievedDoc(self.docs[i][0], self.docs[i][1], float(scores[i]), self.docs[i][2]) for i in order]


def build_retriever(kind: str, corpus_path: str):
    if kind.lower() != "bm25":
        raise NotImplementedError("第一版仅实现 BM25")
    return BM25Retriever(corpus_path)
