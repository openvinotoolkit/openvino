# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify the OV lm_head rebuilds on weight reload/update, instead of going stale.

build_ov_lm_head bakes lm_head.weight into an OV constant at install time
(plugin._install_ov_lm_head). Without the reload_weights/update_weights
patches, any in-place weight mutation after load -- checkpoint reload, RLHF
weight sync -- would leave the compiled lm_head silently serving logits off
the pre-reload weight. This checks both halves: _install_ov_lm_head actually
picks up a mutated weight when re-run, and the two vLLM entry points are
genuinely wrapped rather than silently skipped.

No model download needed: exercises plugin functions against a minimal
stand-in lm_head, and the patch machinery against the real (but
uninitialized) CPUModelRunner/CPUWorker classes.
"""

import os

import pytest

os.environ.setdefault("VLLM_USE_LAYERNAME", "0")

import torch

from openvino.frontend.pytorch.torchdynamo.vllm import plugin


class _FakeLMHead:
    def __init__(self, weight):
        self.weight = weight
        self.cpu_linear = None


class _FakeModel:
    def __init__(self, weight):
        self.lm_head = _FakeLMHead(weight)


def _param(w):
    return torch.nn.Parameter(w.clone(), requires_grad=False)


def _ref(x, w):
    return (x.float() @ w.float().T).to(torch.bfloat16)


@pytest.mark.precommit
def test_install_ov_lm_head_picks_up_reloaded_weight():
    """Re-running _install_ov_lm_head must reflect the new weight, not the old one."""
    torch.manual_seed(0)
    H, V = 8, 32
    w1 = torch.randn(V, H, dtype=torch.bfloat16)
    model = _FakeModel(_param(w1))

    plugin._install_ov_lm_head(model)
    cpu_linear_1 = model.lm_head.cpu_linear
    assert cpu_linear_1 is not None, "OV lm_head failed to compile"

    x = torch.randn(3, H, dtype=torch.bfloat16)
    out1 = cpu_linear_1(x, model.lm_head.weight, None)
    assert torch.allclose(out1.float(), _ref(x, w1).float(), atol=0.1), (
        "OV lm_head output does not match the weight it was built from")

    # Simulate what reload_weights/update_weights do to the parameter.
    w2 = torch.randn(V, H, dtype=torch.bfloat16)
    model.lm_head.weight.data.copy_(w2)

    # Confirm the test is actually exercising staleness: the old closure
    # baked w1 into an OV constant, so it must NOT already track w2.
    stale = cpu_linear_1(x, model.lm_head.weight, None)
    assert not torch.allclose(stale.float(), _ref(x, w2).float(), atol=0.1), (
        "old closure already reflects the new weight -- test setup is wrong")

    # This is what the reload_weights/update_weights patches do afterward.
    plugin._install_ov_lm_head(model)
    cpu_linear_2 = model.lm_head.cpu_linear
    assert cpu_linear_2 is not cpu_linear_1, "lm_head.cpu_linear was not rebuilt"

    out2 = cpu_linear_2(x, model.lm_head.weight, None)
    assert torch.allclose(out2.float(), _ref(x, w2).float(), atol=0.1), (
        "rebuilt OV lm_head does not reflect the reloaded weight")


@pytest.mark.precommit
def test_reload_and_update_weights_are_patched():
    """The two hook points must actually be wrapped, idempotently.

    Force a pristine (unpatched) starting state first: other tests in this
    session build real vLLM engines, which patch these classes for real via
    plugin.register() -- without this reset, this test would see
    already-patched methods and its before/after identity check would be
    meaningless depending on test order.
    """
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner
    from vllm.v1.worker.cpu_worker import CPUWorker

    def _pristine(self, *args, **kwargs):
        pass

    CPUModelRunner.reload_weights = _pristine
    CPUModelRunner._ov_plugin_reload_patched = False
    CPUWorker.update_weights = _pristine
    CPUWorker._ov_plugin_update_patched = False

    orig_reload = CPUModelRunner.reload_weights
    orig_update = CPUWorker.update_weights

    plugin._patch_reload_weights()
    plugin._patch_worker_update_weights()

    assert CPUModelRunner.reload_weights is not orig_reload, (
        "CPUModelRunner.reload_weights was not patched")
    assert CPUWorker.update_weights is not orig_update, (
        "CPUWorker.update_weights was not patched")
    assert getattr(CPUModelRunner, "_ov_plugin_reload_patched", False)
    assert getattr(CPUWorker, "_ov_plugin_update_patched", False)

    # Patching again must not double-wrap.
    once_more = CPUModelRunner.reload_weights
    plugin._patch_reload_weights()
    assert CPUModelRunner.reload_weights is once_more, (
        "re-patching wrapped an already-patched method")


class _FakeCompilationConfig:
    class _Mode:
        name = "STOCK_TORCH_COMPILE"
    mode = _Mode()
    backend = "openvino"


class _FakeVllmConfig:
    compilation_config = _FakeCompilationConfig()


@pytest.mark.precommit
def test_patched_reload_weights_rebuilds_lm_head():
    """Drive the real CPUModelRunner.reload_weights wrapper end to end.

    The original vLLM reload_weights needs a fully loaded engine to run, so
    it is swapped for a minimal stand-in that does the one thing this hook
    depends on: mutate lm_head.weight in place.
    """
    from vllm.v1.worker.cpu_model_runner import CPUModelRunner

    torch.manual_seed(1)
    H, V = 8, 32
    w1 = torch.randn(V, H, dtype=torch.bfloat16)
    model = _FakeModel(_param(w1))
    plugin._install_ov_lm_head(model)
    cpu_linear_1 = model.lm_head.cpu_linear

    w2 = torch.randn(V, H, dtype=torch.bfloat16)

    def _fake_orig_reload_weights(self, *args, **kwargs):
        self.model.lm_head.weight.data.copy_(w2)

    CPUModelRunner.reload_weights = _fake_orig_reload_weights
    CPUModelRunner._ov_plugin_reload_patched = False
    plugin._patch_reload_weights()

    class _FakeRunner:
        vllm_config = _FakeVllmConfig()

        def __init__(self, model):
            self.model = model

    runner = _FakeRunner(model)
    CPUModelRunner.reload_weights(runner)

    assert runner.model.lm_head.cpu_linear is not cpu_linear_1, (
        "reload_weights patch did not rebuild lm_head")
    x = torch.randn(2, H, dtype=torch.bfloat16)
    out = runner.model.lm_head.cpu_linear(x, runner.model.lm_head.weight, None)
    assert torch.allclose(out.float(), _ref(x, w2).float(), atol=0.1), (
        "rebuilt lm_head does not reflect the weight reload_weights wrote")


@pytest.mark.precommit
def test_patched_update_weights_rebuilds_lm_head():
    """Same as above, for CPUWorker.update_weights (RLHF-style weight sync)."""
    from vllm.v1.worker.cpu_worker import CPUWorker

    torch.manual_seed(2)
    H, V = 8, 32
    w1 = torch.randn(V, H, dtype=torch.bfloat16)
    model = _FakeModel(_param(w1))
    plugin._install_ov_lm_head(model)
    cpu_linear_1 = model.lm_head.cpu_linear

    w2 = torch.randn(V, H, dtype=torch.bfloat16)

    def _fake_orig_update_weights(self, *args, **kwargs):
        self.model_runner.get_model().lm_head.weight.data.copy_(w2)

    CPUWorker.update_weights = _fake_orig_update_weights
    CPUWorker._ov_plugin_update_patched = False
    plugin._patch_worker_update_weights()

    class _FakeModelRunner:
        vllm_config = _FakeVllmConfig()

        def __init__(self, model):
            self._model = model

        def get_model(self):
            return self._model

    class _FakeWorker:
        def __init__(self, model_runner):
            self.model_runner = model_runner

    worker = _FakeWorker(_FakeModelRunner(model))
    CPUWorker.update_weights(worker, {})

    assert model.lm_head.cpu_linear is not cpu_linear_1, (
        "update_weights patch did not rebuild lm_head")
    x = torch.randn(2, H, dtype=torch.bfloat16)
    out = model.lm_head.cpu_linear(x, model.lm_head.weight, None)
    assert torch.allclose(out.float(), _ref(x, w2).float(), atol=0.1), (
        "rebuilt lm_head does not reflect the weight update_weights wrote")
