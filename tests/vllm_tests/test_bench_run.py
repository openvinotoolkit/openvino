# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""bench_run must refuse token counts with no steady-state window and never report a non-positive steady time.

No model download needed: the engine setup, _generate and the clock are stubbed.
"""

import types

import pytest

from openvino.frontend.pytorch.torchdynamo.vllm.scripts import bench_run


class _ReachedEngineSetup(Exception):
    pass


def _raise_reached():
    raise _ReachedEngineSetup


def _run_main(monkeypatch, max_new_tokens, skip_warmup_tokens):
    # _select_cpu_platform is the first step after argument parsing.
    monkeypatch.setattr(bench_run, "_select_cpu_platform", _raise_reached)
    monkeypatch.setattr("sys.argv", ["bench_run", "--max-new-tokens", str(max_new_tokens),
                                     "--skip-warmup-tokens", str(skip_warmup_tokens)])
    bench_run.main()


@pytest.mark.precommit
@pytest.mark.parametrize("max_new_tokens, skip_warmup_tokens", [(4, 5), (5, 5), (8, 0), (8, -1)])
def test_invalid_token_counts_rejected(monkeypatch, capsys, max_new_tokens, skip_warmup_tokens):
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, max_new_tokens, skip_warmup_tokens)
    assert exc.value.code == 2
    assert "--skip-warmup-tokens" in capsys.readouterr().err


@pytest.mark.precommit
@pytest.mark.parametrize("max_new_tokens, skip_warmup_tokens", [(6, 5), (64, 5)])
def test_valid_token_counts_accepted(monkeypatch, max_new_tokens, skip_warmup_tokens):
    with pytest.raises(_ReachedEngineSetup):
        _run_main(monkeypatch, max_new_tokens, skip_warmup_tokens)


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
    # Attempt 1: full (1.0 s) not slower than short (1.0 s). Attempt 2: 3.0 - 1.0 = 2.0 s for 6 tokens.
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
