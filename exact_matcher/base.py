from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Protocol


MappingTuple = Tuple[str, str, float]


class ExactMatcher(Protocol):
    name: str

    def run(self, config: Dict[str, Any]) -> List[MappingTuple]:
        ...


@dataclass(frozen=True)
class MatcherResult:
    name: str
    mappings: List[MappingTuple]