from __future__ import annotations

from typing import Any, Dict

from .base import MatcherResult, MappingTuple
from .bertmaplt_string import BERTMapLtStringMatcher
from .logmaplt_file import LogMapLtFileMatcher


_MATCHERS = {
    "bertmaplt_string": BERTMapLtStringMatcher(),
    "logmaplt_file": LogMapLtFileMatcher(),
}


def get_matcher(name: str):
    if name not in _MATCHERS:
        raise ValueError(f"Unknown matcher '{name}'. Available: {sorted(_MATCHERS.keys())}")
    return _MATCHERS[name]


def run_matcher(name: str, config: Dict[str, Any]) -> MatcherResult:
    matcher = get_matcher(name)
    mappings = matcher.run(config)
    return MatcherResult(name=matcher.name, mappings=mappings)