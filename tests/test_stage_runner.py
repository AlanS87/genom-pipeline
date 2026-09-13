"""
Tests for scripts/genom_stage_runner.py -- the officially supported
per-stage/HPC-SLURM driver (one OS process per stage, so the OS/CUDA driver
fully reclaims GPU memory between stages; see the README's "Running
stage-by-stage" section for why this exists alongside run_pipeline()).

These test the driver's own dispatch/config logic (YAML loading, workdir
derivation, the extract overwrite-guard, CLI-vs-YAML equivalence) with the
underlying genom_pipeline.steps.* functions monkeypatched out -- the steps
themselves already have their own real, non-mocked tests (test_extract.py,
test_retrieve.py, test_fusion.py, ...); this file is only about whether the
driver calls them correctly.

scripts/genom_stage_runner.py is a standalone script, not part of the
genom_pipeline package, so it's loaded here via importlib rather than a
normal import.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "genom_stage_runner.py"
_spec = importlib.util.spec_from_file_location("genom_stage_runner", _SCRIPT_PATH)
stage_runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stage_runner
_spec.loader.exec_module(stage_runner)


# ---------------------------------------------------------------------------
# load_yaml_config
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path, text):
    p = tmp_path / "task.yaml"
    p.write_text(text)
    return str(p)


def test_load_yaml_config_fills_defaults_and_applies_stage_section(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        task_name: my_task
        topk: 5

        extract:
          src: /a.owl
          tgt: /b.owl
          reasoner_type: elk
        """,
    )

    config = stage_runner.load_yaml_config(path, "extract")

    assert config["task_name"] == "my_task"
    assert config["topk"] == 5  # shared key
    assert config["src"] == "/a.owl"  # stage-section key
    assert config["reasoner_type"] == "elk"  # stage-section override
    assert config["llm_model"] == stage_runner.DEFAULTS["llm_model"]  # untouched default
    assert config["stage"] == "extract"
    assert config["_config_source"] == str(Path(path).resolve())


def test_load_yaml_config_ignores_other_stage_sections(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        task_name: my_task
        extract:
          src: /a.owl
          tgt: /b.owl
        judge:
          llm_batch_size: 999
        """,
    )
    config = stage_runner.load_yaml_config(path, "extract")
    # judge:'s override must not leak into an extract-stage config
    assert config["llm_batch_size"] == stage_runner.DEFAULTS["llm_batch_size"]


def test_load_yaml_config_rejects_unknown_top_level_key(tmp_path):
    path = _write_yaml(tmp_path, "fuse_strategie: exact_priority\n")  # typo
    with pytest.raises(ValueError, match="unrecognised top-level key"):
        stage_runner.load_yaml_config(path, "fusion")


# ---------------------------------------------------------------------------
# run_stage: workdir derivation
# ---------------------------------------------------------------------------

def test_run_stage_derives_workdir_from_task_name_and_output_root(tmp_path, monkeypatch):
    captured = {}

    def fake_fusion_run(**kwargs):
        captured.update(kwargs)
        return kwargs["output_final_csv"]

    monkeypatch.setattr("genom_pipeline.steps.fusion.run", fake_fusion_run)

    config = dict(stage_runner.DEFAULTS)
    config.update({
        "stage": "fusion",
        "task_name": "my_task",
        "output_root": str(tmp_path / "runs"),
        "_config_source": "cli",
    })
    stage_runner.run_stage(config)

    expected_workdir = str(tmp_path / "runs" / "my_task")
    assert captured["output_final_csv"] == str(Path(expected_workdir) / "final_alignment.csv")


def test_run_stage_explicit_workdir_overrides_task_name(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "genom_pipeline.steps.fusion.run",
        lambda **kw: captured.update(kw) or kw["output_final_csv"],
    )

    explicit_workdir = str(tmp_path / "explicit_dir")
    config = dict(stage_runner.DEFAULTS)
    config.update({
        "stage": "fusion",
        "workdir": explicit_workdir,
        "task_name": "should_be_ignored",
        "output_root": str(tmp_path / "runs"),
        "_config_source": "cli",
    })
    stage_runner.run_stage(config)

    assert captured["output_final_csv"] == str(Path(explicit_workdir) / "final_alignment.csv")


def test_run_stage_missing_both_workdir_and_task_name_raises(tmp_path):
    config = dict(stage_runner.DEFAULTS)
    config.update({"stage": "fusion", "_config_source": "cli"})
    with pytest.raises(ValueError, match="workdir.*task_name"):
        stage_runner.run_stage(config)


# ---------------------------------------------------------------------------
# run_stage: extract's overwrite/skip guard
# ---------------------------------------------------------------------------

def _extract_config(tmp_path, **overrides):
    config = dict(stage_runner.DEFAULTS)
    config.update({
        "stage": "extract",
        "workdir": str(tmp_path),
        "src": "/a.owl",
        "tgt": "/b.owl",
        "_config_source": "cli",
    })
    config.update(overrides)
    return config


def _patch_extract_run(monkeypatch, fake):
    # genom_pipeline/steps/__init__.py does `from .extract import run as
    # extract`, which overwrites the *package's* `extract` attribute with
    # the `run` function itself. So monkeypatch.setattr("genom_pipeline.
    # steps.extract.run", ...) -- which resolves by getattr-chasing through
    # the already-imported `genom_pipeline.steps` package -- finds a
    # function where it expects a module and fails. Patching the real
    # module object (found via import_module(), which resolves through
    # sys.modules by the full dotted name rather than by attribute-chasing)
    # is what run_stage()'s own `from genom_pipeline.steps.extract import
    # run as extract_run` actually resolves against.
    import importlib

    extract_module = importlib.import_module("genom_pipeline.steps.extract")
    monkeypatch.setattr(extract_module, "run", fake)


def test_extract_skips_reparsing_when_concept_store_exists_and_not_overwrite(tmp_path, monkeypatch):
    calls = []
    _patch_extract_run(
        monkeypatch,
        lambda *a, **kw: calls.append((a, kw)) or (tmp_path / "concept_store.pt").as_posix(),
    )

    config = _extract_config(tmp_path, overwrite=False)

    stage_runner.run_stage(config)
    assert len(calls) == 1  # first call: store doesn't exist yet, must run

    (tmp_path / "concept_store.pt").write_bytes(b"fake")
    stage_runner.run_stage(config)
    assert len(calls) == 1, "extract must be skipped once concept_store.pt exists and overwrite=False"

    stage_runner.run_stage(_extract_config(tmp_path, overwrite=True))
    assert len(calls) == 2, "overwrite=True must force re-extraction even if concept_store.pt exists"


def test_extract_forwards_reasoner_type(tmp_path, monkeypatch):
    calls = []
    _patch_extract_run(
        monkeypatch,
        lambda *a, **kw: calls.append(kw) or (tmp_path / "concept_store.pt").as_posix(),
    )

    stage_runner.run_stage(_extract_config(tmp_path, reasoner_type="elk"))

    assert calls[0]["reasoner_type"] == "elk"


# ---------------------------------------------------------------------------
# CLI flags and YAML must resolve to the exact same run_stage() input
# ---------------------------------------------------------------------------

def test_cli_flags_and_yaml_produce_equivalent_fusion_config(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(
        "genom_pipeline.steps.fusion.run",
        lambda **kw: captured.append(kw) or kw["output_final_csv"],
    )

    yaml_path = _write_yaml(
        tmp_path,
        f"""
        workdir: {tmp_path}
        fusion:
          fuse_strategy: exact_priority
          fuse_llm_threshold: 0.9
          fuse_cs_threshold: 0.9
        """,
    )
    yaml_config = stage_runner.load_yaml_config(yaml_path, "fusion")
    stage_runner.run_stage(yaml_config)

    cli_config = dict(stage_runner.DEFAULTS)
    cli_config.update({
        "stage": "fusion",
        "workdir": str(tmp_path),
        "_config_source": "cli",
    })
    stage_runner.run_stage(cli_config)

    assert len(captured) == 2
    yaml_fuse_config, cli_fuse_config = captured[0]["fuse_config"], captured[1]["fuse_config"]
    assert yaml_fuse_config == cli_fuse_config
