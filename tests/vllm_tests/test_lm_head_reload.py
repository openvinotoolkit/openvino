# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The lm_head bakes its weight in at install, so the reload_weights / finish_weight_update patches must rebuild it.

No model download needed: a stand-in lm_head and runner on the real vLLM CPUModelRunner/CPUWorker classes.
"""

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("VLLM_USE_LAYERNAME", "0")

import torch

from openvino.frontend.pytorch.torchdynamo.vllm import plugin

H, V = 8, 32


class _FakeRunner:
    """Stands in for both the model runner and the worker (whose model_runner is itself)."""

    vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(
        mode=SimpleNamespace(name="STOCK_TORCH_COMPILE"), backend="openvino"))

    def __init__(self, weight):
        self.model = SimpleNamespace(lm_head=SimpleNamespace(
            weight=torch.nn.Parameter(weight.clone(), requires_grad=False), cpu_linear=None))
        self.model_runner = self

    def get_model(self):
        return self.model


def _ref(x, w):
    return (x.float() @ w.float().T).to(torch.bfloat16)


def _assert_impl(cpu_linear, impl):
    """A failed OV build falls back to oneDNN silently, so check which one was installed."""
    name = getattr(cpu_linear, "__qualname__", "")
    prefix = "build_ov_lm_head." if impl == "1" else "dispatch_cpu_unquantized_gemm."
    assert name.startswith(prefix), f"expected {prefix}, got {name}"


def _hook(name):
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    from vllm.v1.worker.cpu_worker import CPUWorker

    if name == "reload_weights":
        return CPUModelRunner, "_ov_plugin_reload_patched", plugin._patch_reload_weights
    return CPUWorker, "_ov_plugin_update_patched", plugin._patch_worker_update_weights


@pytest.mark.precommit
@pytest.mark.parametrize("impl", ["1", "0"], ids=["ov_lm_head", "onednn_lm_head"])
@pytest.mark.parametrize("hook", ["reload_weights", "finish_weight_update"])
def test_weight_update_hook_rebuilds_lm_head(hook, impl, monkeypatch):
    monkeypatch.setenv("OV_LM_HEAD", impl)
    cls, flag, patch = _hook(hook)
    torch.manual_seed(0)
    w1, w2 = torch.randn(V, H, dtype=torch.bfloat16), torch.randn(V, H, dtype=torch.bfloat16)
    runner = _FakeRunner(w1)
    lm_head = runner.model.lm_head
    plugin._install_ov_lm_head(runner.model)
    old = lm_head.cpu_linear
    _assert_impl(old, impl)
    x = torch.randn(3, H, dtype=torch.bfloat16)
    assert torch.allclose(old(x, lm_head.weight, None).float(), _ref(x, w1).float(), atol=0.1)

    def fake_original(self, *args, **kwargs):
        self.get_model().lm_head.weight.data.copy_(w2)

    # monkeypatch restores the real method and marker for later tests in the process.
    monkeypatch.setattr(cls, hook, fake_original)
    monkeypatch.setattr(cls, flag, False, raising=False)
    patch()
    patched = getattr(cls, hook)
    assert patched is not fake_original and getattr(cls, flag)
    patch()
    assert getattr(cls, hook) is patched, "re-patching wrapped an already-patched method"

    patched(runner)
    assert not torch.allclose(old(x, lm_head.weight, None).float(), _ref(x, w2).float(), atol=0.1), (
        "the old lm_head already follows the new weight, so this test checks nothing")
    assert lm_head.cpu_linear is not old, f"{hook} patch did not rebuild lm_head"
    _assert_impl(lm_head.cpu_linear, impl)
    assert torch.allclose(lm_head.cpu_linear(x, lm_head.weight, None).float(), _ref(x, w2).float(), atol=0.1)
