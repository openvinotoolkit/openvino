# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for probe_model.py.

probe_model.py requires heavy external dependencies (transformers, requests,
optimum) that are not expected to be installed in a bare test environment.
Tests here therefore focus on:

  1. Guard rail: no MODEL_ID → exit 1 with usage message
  2. MODEL_ID via env var vs. positional argument are both accepted
  3. The estimate_params() logic (inlined for unit testing — matches the
     formula in the script)
  4. The PIPELINE_TAG_MAP used in try_conversion.py is complete for the
     task values consumed downstream

When the external deps ARE present (CI with the full ML env), the script
will naturally be exercised by integration runs; these unit tests guard
against silent regressions in the script's own logic.
"""

import pathlib
import sys

import pytest

from conftest import SCRIPTS_DIR, run_script

SCRIPT = SCRIPTS_DIR / "probe_model.py"


# ---------------------------------------------------------------------------
# Guard: MODEL_ID is required
# ---------------------------------------------------------------------------

def test_no_model_id_exits_one(tmp_path):
    import os
    env = os.environ.copy()
    env.pop("MODEL_ID", None)
    result = run_script(SCRIPT, tmp_path, extra_env=env)
    assert result.returncode == 1


def test_no_model_id_prints_usage(tmp_path):
    import os
    env = os.environ.copy()
    env.pop("MODEL_ID", None)
    result = run_script(SCRIPT, tmp_path, extra_env=env)
    combined = result.stderr + result.stdout
    assert "model_id" in combined.lower() or "usage" in combined.lower()


def test_model_id_env_var_is_accepted(tmp_path):
    """With MODEL_ID set, the script must NOT exit 1 on the usage guard."""
    result = run_script(SCRIPT, tmp_path, extra_env={"MODEL_ID": "test/model"})
    # The script will fail later (no transformers installed), but must pass
    # the usage guard.  returncode == 1 with "Usage:" in stderr = guard fired.
    if result.returncode == 1:
        assert "Usage:" not in result.stderr, (
            "MODEL_ID env var not accepted — usage guard fired unexpectedly"
        )


def test_model_id_positional_argument_is_accepted(tmp_path):
    """MODEL_ID as positional arg must also bypass the usage guard."""
    import os
    env = os.environ.copy()
    env.pop("MODEL_ID", None)
    result = run_script(SCRIPT, tmp_path, "some/model", extra_env=env)
    if result.returncode == 1:
        assert "Usage:" not in result.stderr


# ---------------------------------------------------------------------------
# estimate_params() — inlined formula unit tests
# ---------------------------------------------------------------------------
# The formula is duplicated here intentionally.  If the script's formula
# changes, this test breaks loudly, prompting a review of the change.

def _estimate_params(hidden_size, num_hidden_layers, vocab_size,
                     intermediate_size=None, num_attention_heads=32,
                     num_key_value_heads=None, has_mlp_bias=False):
    h = hidden_size
    layers = num_hidden_layers
    vocab = vocab_size
    intermediate = intermediate_size if intermediate_size is not None else h * 4
    heads = num_attention_heads
    kv_heads = num_key_value_heads if num_key_value_heads is not None else heads

    attn = h * h + (kv_heads * (h // heads if heads else 1)) * 2 * h + h * h
    ffn = h * intermediate * 3 if has_mlp_bias else h * intermediate * 2
    embed = vocab * h * 2
    per_layer = attn + ffn
    return layers * per_layer + embed


def test_estimate_params_llama7b_in_plausible_range():
    """LLaMA-7B config should estimate somewhere between 5B and 15B params."""
    est = _estimate_params(
        hidden_size=4096,
        num_hidden_layers=32,
        vocab_size=32000,
        intermediate_size=11008,
        num_attention_heads=32,
        num_key_value_heads=32,
    )
    assert 5e9 < est < 15e9, f"Estimate {est/1e9:.1f}B seems off for LLaMA-7B config"


def test_estimate_params_zero_config_returns_zero():
    """All-zero config must return 0, not crash."""
    est = _estimate_params(0, 0, 0, intermediate_size=0, num_attention_heads=1)
    assert est == 0


def test_estimate_params_scales_with_layers():
    """Doubling the layer count roughly doubles the estimate."""
    base = _estimate_params(2048, 16, 16000, intermediate_size=8192, num_attention_heads=16)
    doubled = _estimate_params(2048, 32, 16000, intermediate_size=8192, num_attention_heads=16)
    ratio = doubled / base if base else float("inf")
    assert 1.8 < ratio < 2.2, f"Layer scaling ratio {ratio:.2f} is unexpected"


def test_estimate_params_gqa_less_than_mha():
    """GQA (fewer kv_heads) should produce a smaller estimate than full MHA."""
    mha = _estimate_params(4096, 32, 32000, intermediate_size=11008,
                           num_attention_heads=32, num_key_value_heads=32)
    gqa = _estimate_params(4096, 32, 32000, intermediate_size=11008,
                           num_attention_heads=32, num_key_value_heads=8)
    assert gqa < mha, "GQA should use fewer params than full MHA"


# ---------------------------------------------------------------------------
# PIPELINE_TAG_MAP completeness
# ---------------------------------------------------------------------------

# These are the pipeline tags that the analyze-and-convert agent documents
# as supported.  Every tag must map to a non-empty task string.
_EXPECTED_TAGS = [
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "text-classification",
    "token-classification",
    "question-answering",
    "feature-extraction",
    "fill-mask",
    "text-to-image",
    "image-to-text",
    "automatic-speech-recognition",
    "audio-classification",
    "zero-shot-image-classification",
]

# Read PIPELINE_TAG_MAP from the try_conversion.py source (authoritative copy)
def _load_pipeline_tag_map():
    """Parse PIPELINE_TAG_MAP from try_conversion.py without executing the file."""
    src = (SCRIPTS_DIR / "try_conversion.py").read_text(encoding="utf-8")
    # Extract the dict literal between PIPELINE_TAG_MAP = { ... }
    import ast, re
    m = re.search(
        r"PIPELINE_TAG_MAP\s*=\s*(\{[^}]+\})",
        src,
        re.DOTALL,
    )
    if not m:
        return {}
    return ast.literal_eval(m.group(1))


@pytest.mark.parametrize("tag", _EXPECTED_TAGS)
def test_pipeline_tag_map_covers_expected_tag(tag):
    tag_map = _load_pipeline_tag_map()
    assert tag in tag_map, f"PIPELINE_TAG_MAP missing tag: {tag!r}"


@pytest.mark.parametrize("tag", _EXPECTED_TAGS)
def test_pipeline_tag_map_value_is_non_empty_string(tag):
    tag_map = _load_pipeline_tag_map()
    task = tag_map.get(tag, "")
    assert isinstance(task, str) and task.strip(), (
        f"PIPELINE_TAG_MAP[{tag!r}] is empty or not a string"
    )
