from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

MappingTuple = Tuple[str, str, float]


@dataclass
class LogMapLtFileMatcher:
    name: str = "logmaplt_file"

    def run(self, config: Dict[str, Any]) -> List[MappingTuple]:
        tsv_path = config["mapping_tsv"]
        sep = config.get("sep", "\t")
        has_header = bool(config.get("has_header", False))

        if has_header:
            df = pd.read_csv(tsv_path, sep=sep)
            cols = config.get("columns", ["src", "tgt", "score"])
            src_col, tgt_col, score_col = cols[0], cols[1], cols[2]
            rows = df[[src_col, tgt_col, score_col]].itertuples(index=False, name=None)
        else:
            df = pd.read_csv(tsv_path, sep=sep, header=None)
            rows = df.itertuples(index=False, name=None)

        out: List[MappingTuple] = []
        for r in rows:
            src = str(r[0])
            tgt = str(r[1])
            score = float(r[2]) if len(r) > 2 else 1.0
            out.append((src, tgt, score))
        return out