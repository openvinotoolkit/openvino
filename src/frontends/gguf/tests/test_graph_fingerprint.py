# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Gated graph-fingerprint regression test for the GGUF frontend native builder.
#
# The GGUF files are multi-GB and cannot ship in-repo, so this test is OPT-IN: it runs only when
# GGUF_FINGERPRINT_MODELS points at local models. It converts each model through the frontend
# (requested by name, since it is not auto-selectable) and asserts the graph fingerprint matches
# any unintended change to the produced graph for a supported architecture.
#
# GGUF_FINGERPRINT_MODELS format: semicolon-separated entries "name=path=expected_sighash"
# (the expected hash is optional; when omitted the test just asserts the model converts and prints
# the observed hash so it can be recorded as the baseline).
#
# Example:
#   GGUF_FINGERPRINT_MODELS="qwen3=/models/Qwen3-0.6B-Q8_0.gguf=87cf7f3b0742a6cd" \
#   PYTHONPATH=<ov>/python LD_LIBRARY_PATH=<ov> pytest test_graph_fingerprint.py -v

import os

import pytest

from graph_fingerprint import fingerprint

_SPEC = os.environ.get("GGUF_FINGERPRINT_MODELS", "").strip()


def _parse_spec(spec):
    cases = []
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split("=")
        name, path = parts[0], parts[1]
        expected = parts[2] if len(parts) > 2 else None
        cases.append((name, path, expected))
    return cases


_CASES = _parse_spec(_SPEC)


@pytest.mark.skipif(not _CASES, reason="set GGUF_FINGERPRINT_MODELS=name=path[=hash];... to run")
@pytest.mark.parametrize("name,path,expected", _CASES, ids=[c[0] for c in _CASES])
def test_graph_fingerprint(name, path, expected):
    assert os.path.exists(path), f"model not found: {path}"
    fp = fingerprint(path)
    print(f"\n{name}: sighash={fp['sighash']} ops={fp['num_ops']} inputs={fp['inputs']}")
    if expected:
        assert fp["sighash"] == expected, (
            f"{name} graph fingerprint changed: got {fp['sighash']}, expected {expected}. "
            f"The builder produced a different graph for this architecture."
        )
