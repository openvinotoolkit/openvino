# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify bench_run._run never reports a throughput from a non-positive steady-state time.

The steady-state time is the full run minus the short run. Timing noise can make it zero or negative; _run used to
clamp it to 1 ns and report an enormous tok/s. It must re-measure, and fail if no attempt is valid.

No model download needed: _generate and the clock are stubbed.
"""

import types

import pytest

from openvino.frontend.pytorch.torchdynamo.vllm.scripts import bench_run


def _stub(monkeypatch, durations):
    """Each _generate call after the warmup takes the next duration from `durations`."""
    calls = []
    clock = [0.0]
    pending = list(durations)

    def generate(llm, prompt, params):
        calls.append(params.max_tokens)
        if len(calls) > 1:  # the first call is the untimed warmup
            clock[0] += pending.pop(0)
        return types.SimpleNamespace(text="out")

    monkeypatch.setattr(bench_run, "_generate", generate)
    monkeypatch.setattr(bench_run.time, "perf_counter", lambda: clock[0])
    return calls


@pytest.mark.precommit
def test_non_positive_steady_time_is_remeasured(monkeypatch):
    # Attempt 1: full (1.0 s) not slower than short (1.0 s). Attempt 2: valid, 3.0 - 1.0 = 2.0 s for 6 tokens.
    calls = _stub(monkeypatch, [1.0, 1.0, 1.0, 3.0])
    text, tps = bench_run._run("x", llm=None, prompt="p", max_new_tokens=8, skip_warmup_tokens=2)
    assert text == "out"
    assert tps == pytest.approx(6 / 2.0)
    assert calls == [8, 2, 8, 2, 8]


@pytest.mark.precommit
def test_never_positive_steady_time_raises(monkeypatch):
    calls = _stub(monkeypatch, [2.0, 1.0] * bench_run._MEASURE_ATTEMPTS)
    with pytest.raises(RuntimeError, match="not slower"):
        bench_run._run("x", llm=None, prompt="p", max_new_tokens=8, skip_warmup_tokens=2)
    assert len(calls) == 1 + 2 * bench_run._MEASURE_ATTEMPTS
