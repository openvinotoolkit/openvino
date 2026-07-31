# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for classify_failure.py.

Covers:
  - All 11 error taxonomy classes
  - Dominant-class selection (last failed attempt)
  - Routing decisions for every taxonomy class
  - OOM, custom-ops, VLM signal extraction
  - transformers_override signal
  - Empty / all-success attempts (safe fallback)
  - Missing input files (no crash)
  - error_excerpts.json correctness
  - Excerpt starts at last Traceback block
"""

import json
import pathlib

import pytest

from conftest import SCRIPTS_DIR, run_script, write_json

SCRIPT = SCRIPTS_DIR / "classify_failure.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attempt(aid, success, stderr="", stdout=""):
    return {"id": aid, "success": success, "stderr": stderr, "stdout": stdout}


def _profile(is_vlm=False, special_config_keys=None, trust_remote_code_required=False):
    return {
        "model_id": "test/model",
        "trust_remote_code_required": trust_remote_code_required,
        "is_vlm": is_vlm,
        "special_config_keys": special_config_keys or [],
    }


def _run(tmp_path, attempts, profile=None):
    write_json(tmp_path / "conversion_attempts.json", attempts)
    write_json(tmp_path / "model_profile.json", profile or _profile())
    return run_script(SCRIPT, tmp_path)


def _signals(tmp_path):
    return json.loads((tmp_path / "routing_signals.json").read_text(encoding="utf-8"))


def _excerpts(tmp_path):
    return json.loads((tmp_path / "error_excerpts.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Error taxonomy — one parametrised test per class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expected_class, error_text", [
    (
        "missing_model_dependency",
        "ModuleNotFoundError: No module named 'flash_attn'",
    ),
    (
        "missing_model_dependency",
        "ImportError: requires package 'einops'. Install via pip.",
    ),
    (
        "optimum_unsupported_arch",
        "KeyError: 'TasksManager' model_type not support llama4",
    ),
    (
        "unknown_arch_transformers_too_old",
        # Must match outer (no configuration class) AND inner (transformers doesn't know)
        "no configuration class found for model_type and transformers doesn't know this arch",
    ),
    (
        "optimum_export_bug",
        "Exception raised in optimum/exporters/openvino dummy_inputs TypeError",
    ),
    (
        "missing_conversion_rule",
        "NotImplementedError: aten::erfinv — no rule for this op",
    ),
    (
        "frontend_error",
        "Exception in openvino/frontend/pytorch ir parse error",
    ),
    (
        "ir_validation_error",
        "ngraph::validate_and_infer_types shape inference failed",
    ),
    (
        "inference_runtime_error",
        "ov::exception raised in infer request plugin error openvino runtime",
    ),
    (
        "genai_unsupported",
        "ValueError from openvino_genai chat template missing pipeline construct failed",
    ),
    (
        "tokenizer_error",
        "openvino_tokenizers sentencepiece tokenizer convert error",
    ),
    (
        "unknown",
        "some completely unrecognised xyzzy error that matches nothing",
    ),
])
def test_error_class_classification(tmp_path, expected_class, error_text):
    """Every error pattern maps to the correct taxonomy class."""
    result = _run(tmp_path, [_attempt("A", False, stderr=error_text)])
    assert result.returncode == 0, result.stderr
    assert _signals(tmp_path)["error_class"] == expected_class


# ---------------------------------------------------------------------------
# Dominant class = last failed attempt
# ---------------------------------------------------------------------------

def test_dominant_class_taken_from_last_failed_attempt(tmp_path):
    attempts = [
        _attempt("A", False, stderr="ModuleNotFoundError no module named x"),
        _attempt("B", False, stderr="ov::exception infer request plugin error"),
    ]
    _run(tmp_path, attempts)
    # B is the last → inference_runtime_error, not missing_model_dependency
    assert _signals(tmp_path)["error_class"] == "inference_runtime_error"


def test_successful_attempt_ignored_in_classification(tmp_path):
    """A trailing successful attempt must not change the dominant class."""
    attempts = [
        _attempt("A", False, stderr="ov::exception infer request"),
        _attempt("B", True, stdout="Export done"),  # success — skip
    ]
    _run(tmp_path, attempts)
    assert _signals(tmp_path)["error_class"] == "inference_runtime_error"


# ---------------------------------------------------------------------------
# Routing decisions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error_class_text, expected_agent", [
    ("ModuleNotFoundError no module named foo",               "optimum-intel"),
    ("KeyError TasksManager model_type not support",          "optimum-intel"),
    ("no class for model_type transformers doesn't know",     "optimum-intel"),
    ("Exception in optimum/exporters dummy_inputs TypeError", "optimum-intel"),
    ("some completely unrecognised error",                    "optimum-intel"),
    ("NotImplementedError aten::erfinv no rule",              "enable-operator"),
    ("openvino/frontend pytorch ir parse error",              "enable-operator"),
    ("validate_and_infer_types shape inference failed",       "enable-operator"),
    ("ov::exception infer request plugin error",              "enable-operator"),
    ("openvino_genai chat template missing",                  "openvino-genai"),
    ("openvino_tokenizers sentencepiece convert",             "openvino-tokenizers"),
])
def test_routing_target_agent(tmp_path, error_class_text, expected_agent):
    _run(tmp_path, [_attempt("A", False, stderr=error_class_text)])
    assert _signals(tmp_path)["target_agent"] == expected_agent


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def test_oom_detected_in_stderr(tmp_path):
    err = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
    _run(tmp_path, [_attempt("A", False, stderr=err)])
    assert _signals(tmp_path)["oom_suspected"] is True


def test_oom_detected_in_stdout(tmp_path):
    """OOM in stdout (some backends print there) must also be caught."""
    err = "Killed — memory error allocating buffer"
    _run(tmp_path, [_attempt("A", False, stdout=err)])
    assert _signals(tmp_path)["oom_suspected"] is True


def test_custom_ops_detected_via_ssm_config_keys(tmp_path):
    profile = _profile(special_config_keys=["ssm_state_size", "recurrent_layer"])
    _run(tmp_path, [_attempt("A", False, stderr="unrelated")], profile=profile)
    assert _signals(tmp_path)["custom_ops_suspected"] is True


def test_mamba_key_triggers_custom_ops(tmp_path):
    profile = _profile(special_config_keys=["mamba_d_state"])
    _run(tmp_path, [_attempt("A", False, stderr="unrelated")], profile=profile)
    assert _signals(tmp_path)["custom_ops_suspected"] is True


def test_non_ssm_keys_do_not_trigger_custom_ops(tmp_path):
    profile = _profile(special_config_keys=["hidden_size", "num_layers"])
    _run(tmp_path, [_attempt("A", False, stderr="unrelated")], profile=profile)
    assert _signals(tmp_path)["custom_ops_suspected"] is False


def test_vlm_sets_tokenizer_check_signal(tmp_path):
    profile = _profile(is_vlm=True)
    _run(tmp_path, [_attempt("A", False, stderr="unrelated")], profile=profile)
    assert _signals(tmp_path)["requires_tokenizer_check"] is True


def test_non_vlm_does_not_set_tokenizer_check(tmp_path):
    _run(tmp_path, [_attempt("A", False, stderr="unrelated")])
    assert _signals(tmp_path)["requires_tokenizer_check"] is False


def test_optimum_new_arch_signal_set(tmp_path):
    err = "KeyError: TasksManager model_type not support"
    _run(tmp_path, [_attempt("A", False, stderr=err)])
    assert _signals(tmp_path)["requires_optimum_new_arch"] is True


def test_transformers_too_old_sets_upgrade_and_override(tmp_path):
    err = "no configuration class found for model_type and transformers doesn't know this arch"
    _run(tmp_path, [_attempt("A", False, stderr=err)])
    s = _signals(tmp_path)
    assert s["requires_transformers_upgrade"] is True
    assert "github.com/huggingface/transformers" in s["transformers_override"]


def test_trust_remote_code_propagated_from_profile(tmp_path):
    profile = _profile(trust_remote_code_required=True)
    _run(tmp_path, [_attempt("A", False, stderr="unrelated")], profile=profile)
    assert _signals(tmp_path)["trust_remote_code_required"] is True


# ---------------------------------------------------------------------------
# Edge cases: empty / all-success / missing files
# ---------------------------------------------------------------------------

def test_empty_attempts_produces_unknown_fallback(tmp_path):
    _run(tmp_path, [])
    s = _signals(tmp_path)
    assert s["error_class"] == "unknown"
    assert s["target_agent"] == "optimum-intel"


def test_all_successful_attempts_produces_unknown_fallback(tmp_path):
    """Zero failed attempts → no text to classify → unknown is the safe default."""
    attempts = [_attempt("A", True, stdout="OK"), _attempt("B", True, stdout="OK")]
    _run(tmp_path, attempts)
    assert _signals(tmp_path)["error_class"] == "unknown"


def test_missing_profile_does_not_crash(tmp_path):
    write_json(tmp_path / "conversion_attempts.json", [_attempt("A", False, stderr="ov::exception")])
    # no model_profile.json
    result = run_script(SCRIPT, tmp_path)
    assert result.returncode == 0
    assert (tmp_path / "routing_signals.json").exists()


def test_missing_attempts_does_not_crash(tmp_path):
    write_json(tmp_path / "model_profile.json", _profile())
    # no conversion_attempts.json
    result = run_script(SCRIPT, tmp_path)
    assert result.returncode == 0
    assert (tmp_path / "routing_signals.json").exists()


def test_missing_both_inputs_does_not_crash(tmp_path):
    result = run_script(SCRIPT, tmp_path)
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Output: routing_signals.json schema
# ---------------------------------------------------------------------------

def test_routing_signals_has_all_required_keys(tmp_path):
    _run(tmp_path, [_attempt("A", False, stderr="unrelated")])
    s = _signals(tmp_path)
    required = {
        "error_class", "target_agent",
        "requires_optimum_new_arch", "requires_transformers_upgrade",
        "transformers_override", "requires_tokenizer_check",
        "is_vlm", "custom_ops_suspected", "oom_suspected",
        "trust_remote_code_required",
    }
    missing = required - set(s.keys())
    assert not missing, f"routing_signals.json missing keys: {missing}"


# ---------------------------------------------------------------------------
# Output: error_excerpts.json correctness
# ---------------------------------------------------------------------------

def test_error_excerpts_written_for_each_failed_attempt(tmp_path):
    attempts = [
        _attempt("A", False, stderr="Traceback (most recent call last):\nValueError: bad"),
        _attempt("B", False, stderr="RuntimeError: ov::exception"),
    ]
    _run(tmp_path, attempts)
    ex = _excerpts(tmp_path)
    assert "A" in ex and "B" in ex


def test_excerpt_starts_at_last_traceback_block(tmp_path):
    """Excerpts must begin at 'Traceback (most recent call last)' — not at INFO lines."""
    noise = "\n".join(f"INFO step {i}" for i in range(40))
    tb = "Traceback (most recent call last):\n  File x.py, line 1\nRuntimeError: real error"
    _run(tmp_path, [_attempt("A", False, stderr=noise + "\n" + tb)])
    ex = _excerpts(tmp_path)
    assert ex["A"].startswith("Traceback"), (
        f"Excerpt should start with Traceback, got: {ex['A'][:60]!r}"
    )


def test_excerpt_not_present_for_successful_attempt(tmp_path):
    attempts = [_attempt("OK", True, stdout="done")]
    _run(tmp_path, attempts)
    ex = _excerpts(tmp_path)
    assert "OK" not in ex
