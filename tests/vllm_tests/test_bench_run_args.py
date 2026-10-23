# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify bench_run rejects token counts that leave no steady-state window.

With --skip-warmup-tokens >= --max-new-tokens, _run() reported zero or
negative throughput; with --skip-warmup-tokens 0, vLLM rejected the short
run's max_tokens=0, but only after the model had loaded. main() must
refuse these through argparse before touching any engine.

No model download needed: _select_cpu_platform (the first step after
argument parsing) is replaced by a sentinel, so valid arguments stop there.
"""

import pytest

from openvino.frontend.pytorch.torchdynamo.vllm.scripts import bench_run


class _ReachedEngineSetup(Exception):
    pass


def _raise_reached():
    raise _ReachedEngineSetup


def _run_main(monkeypatch, max_new_tokens, skip_warmup_tokens):
    monkeypatch.setattr(bench_run, "_select_cpu_platform", _raise_reached)
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_run",
            "--max-new-tokens",
            str(max_new_tokens),
            "--skip-warmup-tokens",
            str(skip_warmup_tokens),
        ],
    )
    bench_run.main()


@pytest.mark.precommit
@pytest.mark.parametrize(
    "max_new_tokens, skip_warmup_tokens",
    [(4, 5), (5, 5), (8, 0), (8, -1)],
)
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
