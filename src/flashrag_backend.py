"""Optional FlashRAG component backend.

The pipeline only relies on ``search`` and ``generate``; this adapter keeps the
fixed-hop controller independent of FlashRAG version-specific APIs.
"""


class FlashRAGRetriever:
    def __init__(self, config: dict):
        from flashrag.config import Config
        from flashrag.utils import get_retriever

        self._retriever = get_retriever(Config(config_dict=config))

    def search(self, query: str, top_k: int = 5):
        result = self._retriever.search(query, num=top_k)
        from .schema import RetrievedDoc

        return [
            RetrievedDoc(
                str(item.get("id", item.get("doc_id", index))),
                item.get("contents", item.get("text", "")),
                float(item.get("score", 0.0)),
                item,
            )
            for index, item in enumerate(result)
        ]


class FlashRAGGenerator:
    def __init__(self, config: dict):
        from flashrag.config import Config
        from flashrag.utils import get_generator

        self._generator = get_generator(Config(config_dict=config))

    def generate(self, prompt: str) -> str:
        result = self._generator.generate([prompt])
        return result[0] if isinstance(result, list) else str(result)
