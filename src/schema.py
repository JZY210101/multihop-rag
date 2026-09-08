from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class Sample:
    sample_id: str
    question: str
    answer: str = ""
    hop_num: int = 2
    supporting_facts: List[Any] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
