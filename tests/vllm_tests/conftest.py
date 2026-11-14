# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the vLLM + OpenVINO torchdynamo backend test suite."""

import gc
import os

import pytest

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# The tests ship plain functions through collective_rpc (local test IPC only).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

# Must precede the first `import vllm`: vLLM reads it at import, before the plugin loads.
from openvino.frontend.pytorch.torchdynamo.vllm.preset import (  # noqa: E402
    set_pre_import_env,
)

set_pre_import_env()

pytest.importorskip("vllm")

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def pytest_configure(config):
    config.addinivalue_line("markers", "precommit: Tests to run on pre-commit CI")
    config.addinivalue_line("markers", "nightly: Tests to run on nightly CI")


@pytest.fixture(autouse=True)
def _gc_between_tests():
    """Collect after every test, so one test's engine isn't still resident in the next."""
    yield
    gc.collect()


def select_cpu_platform():
    """Force the CPU platform: with both `vllm` and `vllm-cpu` installed, auto-detection can pick the wrong one."""
    import vllm.platforms as _vp
    from vllm.platforms.cpu import CpuPlatform as _CpuPlatform
    _vp._current_platform = _CpuPlatform()


def new_openvino_llm(**overrides):
    """Build a bf16 OV-backend TinyLlama; each test builds and deletes its own, as a shared one OOMed CI.

    block_size=32 is required by the OV CPU PagedAttention kernel.
    """
    select_cpu_platform()
    from vllm import LLM

    kwargs = dict(
        model=MODEL_ID,
        dtype="bfloat16",
        enforce_eager=False,
        max_model_len=2048,
        distributed_executor_backend="uni",
        block_size=32,
        compilation_config={
            "mode": "STOCK_TORCH_COMPILE",
            "backend": "openvino",
            "custom_ops": ["none"],
        },
    )
    kwargs.update(overrides)
    return LLM(**kwargs)
