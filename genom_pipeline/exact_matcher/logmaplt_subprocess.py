from __future__ import annotations

"""
Runs LogMap's LITE mode as an external Java process and parses its output.

This is the "actually run LogMap" counterpart to logmaplt_file.py, which only
*loads* a mapping TSV that was produced by some earlier, separate LogMap run.
Use logmaplt_subprocess when you want the pipeline itself to invoke LogMap;
use logmaplt_file when a mapping file already exists (e.g. produced by a
teammate, or by a prior manual run) and you just want to feed it into fusion.

LogMap itself is NOT vendored in this repository -- it is a large (tens of MB)
third-party Java tool with its own release cycle. See scripts/download_logmap.sh
to fetch the official standalone distribution instead of committing it to git.
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

MappingTuple = Tuple[str, str, float]

# LogMap's own recommended JVM flags (see the project README). max heap is
# intentionally conservative here -- override via config["jvm_args"] for
# large ontologies.
DEFAULT_JVM_ARGS = [
    "-Xms500M",
    "-Xmx8G",
    "-DentityExpansionLimit=10000000",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
]


def _resolve_jar_path(config: Dict[str, Any]) -> Path:
    jar_path = config.get("jar_path") or os.getenv("LOGMAP_JAR_PATH")
    if not jar_path:
        raise ValueError(
            "No LogMap jar path given. Pass exact_config['jar_path'], or set the "
            "LOGMAP_JAR_PATH environment variable, after fetching the jar with "
            "scripts/download_logmap.sh (LogMap is not vendored in this repo)."
        )
    path = Path(jar_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"LogMap jar not found at {path}. Run scripts/download_logmap.sh, "
            "or point jar_path / LOGMAP_JAR_PATH at your own copy."
        )
    return path


def _run_logmap_lite(
    jar_path: Path,
    src_owl: str,
    tgt_owl: str,
    output_dir: Path,
    java_bin: str,
    jvm_args: List[str],
    timeout: int,
) -> subprocess.CompletedProcess:
    output_dir.mkdir(parents=True, exist_ok=True)

    src_uri = Path(src_owl).expanduser().resolve().as_uri()
    tgt_uri = Path(tgt_owl).expanduser().resolve().as_uri()
    # LogMap's CLI wants a trailing separator on the output path.
    output_dir_arg = str(output_dir) + os.sep

    cmd = [java_bin, *jvm_args, "-jar", str(jar_path), "LITE", src_uri, tgt_uri, output_dir_arg]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            "LogMap LITE failed "
            f"(exit code {result.returncode}).\ncmd: {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def _find_mapping_output(
    output_dir: Path,
    files_before: set,
    mapping_filename: Optional[str],
) -> Path:
    """
    LogMap LITE's exact output filename is an internal implementation detail
    we don't want to hard-code a guess for (it may differ across LogMap
    versions). Two ways to resolve it:
      1. Caller passes mapping_filename explicitly (recommended once you've
         confirmed it for your LogMap version -- check output_dir after a
         first manual run).
      2. Otherwise, diff output_dir's contents before/after the subprocess
         call and use whichever single new tabular file (.tsv/.txt/.csv)
         appeared.
    """
    if mapping_filename:
        candidate = output_dir / mapping_filename
        if not candidate.exists():
            raise FileNotFoundError(f"Expected LogMap output file not found: {candidate}")
        return candidate

    files_after = set(output_dir.iterdir())
    new_files = [p for p in (files_after - files_before) if p.is_file()]
    mapping_like = [p for p in new_files if p.suffix.lower() in (".tsv", ".txt", ".csv")]

    if len(mapping_like) == 1:
        return mapping_like[0]

    if len(mapping_like) == 0:
        raise RuntimeError(
            f"LogMap ran successfully but no new .tsv/.txt/.csv file appeared in "
            f"{output_dir}. Files now present: {sorted(p.name for p in files_after)}. "
            "Pass exact_config['mapping_filename'] explicitly once you know which "
            "file LogMap produced on your machine/version."
        )

    raise RuntimeError(
        f"LogMap produced multiple new candidate files in {output_dir}: "
        f"{sorted(p.name for p in mapping_like)}. Pass exact_config['mapping_filename'] "
        "to disambiguate which one holds the equivalence mappings."
    )


def _parse_logmap_output(path: Path, sep: str, has_header: bool) -> List[MappingTuple]:
    if has_header:
        df = pd.read_csv(path, sep=sep)
        cols = list(df.columns[:3])
        rows = df[cols].itertuples(index=False, name=None)
    else:
        df = pd.read_csv(path, sep=sep, header=None)
        rows = df.itertuples(index=False, name=None)

    out: List[MappingTuple] = []
    for r in rows:
        src = str(r[0])
        tgt = str(r[1])
        score = float(r[2]) if len(r) > 2 else 1.0
        out.append((src, tgt, score))
    return out


@dataclass
class LogMapLtSubprocessMatcher:
    name: str = "logmaplt_subprocess"

    def run(self, config: Dict[str, Any]) -> List[MappingTuple]:
        """
        config keys
          src_owl, tgt_owl: paths to the source/target .owl files (required)
          jar_path: path to logmap-matcher-*.jar. Falls back to the
            LOGMAP_JAR_PATH environment variable.
          output_dir: directory LogMap writes its output into. Default
            "logmap_lite_output" relative to the current working directory --
            pass an explicit per-task path (e.g. workdir/"logmap") so
            concurrent tasks don't clobber each other.
          java_bin: default "java"
          jvm_args: default DEFAULT_JVM_ARGS
          timeout: seconds, default 1800
          mapping_filename: explicit output filename inside output_dir. See
            _find_mapping_output for why this isn't hard-coded by default.
          sep: separator for the output file, default "\\t"
          has_header: whether the output file has a header row, default False
          overwrite: bool, default False. If True, re-run LogMap even when
            mapping_filename already exists in output_dir.
        """
        if "src_owl" not in config or "tgt_owl" not in config:
            raise ValueError("logmaplt_subprocess requires config['src_owl'] and config['tgt_owl'].")

        jar_path = _resolve_jar_path(config)
        output_dir = Path(config.get("output_dir", "logmap_lite_output")).expanduser().resolve()
        java_bin = config.get("java_bin", "java")
        jvm_args = list(config.get("jvm_args", DEFAULT_JVM_ARGS))
        timeout = int(config.get("timeout", 1800))
        mapping_filename = config.get("mapping_filename")
        sep = config.get("sep", "\t")
        has_header = bool(config.get("has_header", False))
        overwrite = bool(config.get("overwrite", False))

        output_dir.mkdir(parents=True, exist_ok=True)

        if mapping_filename and not overwrite:
            candidate = output_dir / mapping_filename
            if candidate.exists():
                return _parse_logmap_output(candidate, sep=sep, has_header=has_header)

        files_before = set(output_dir.iterdir())
        _run_logmap_lite(
            jar_path=jar_path,
            src_owl=config["src_owl"],
            tgt_owl=config["tgt_owl"],
            output_dir=output_dir,
            java_bin=java_bin,
            jvm_args=jvm_args,
            timeout=timeout,
        )
        mapping_path = _find_mapping_output(output_dir, files_before, mapping_filename)

        return _parse_logmap_output(mapping_path, sep=sep, has_header=has_header)
