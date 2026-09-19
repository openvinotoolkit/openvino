# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the vLLM + OpenVINO torchdynamo backend test suite."""

import os

import pytest

pytest.importorskip("vllm")

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

# Must precede the first `import vllm`: vLLM reads this at module import,
# well before the plugin entry point loads. See vllm.preset.set_pre_import_env.
from openvino.frontend.pytorch.torchdynamo.vllm.preset import (  # noqa: E402
    set_pre_import_env,
)

set_pre_import_env()

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def pytest_configure(config):
    config.addinivalue_line("markers", "precommit: Tests to run on pre-commit CI")
    config.addinivalue_line("markers", "nightly: Tests to run on nightly CI")


def select_cpu_platform():
    """Force CPU platform: this env may have both `vllm` and `vllm-cpu` installed.

    Auto-detection can pick the wrong one, so pre-init the platform to CPU
    before any vLLM engine is built. Mirrors the standard CPU-only vLLM
    bootstrapping documented in vllm/getting_started/installation/cpu/.
    """
    import vllm.platforms as _vp
    from vllm.platforms.cpu import CpuPlatform as _CpuPlatform
    _vp._current_platform = _CpuPlatform()


@pytest.fixture(scope="session")
def openvino_llm():
    """A single OV-backend TinyLlama LLM instance, shared across the session.

    OV CPU PagedAttention requires block_size=32 (hard kernel constraint).
    custom_ops=["none"] keeps vLLM from expanding RMSNorm/SiLU into custom
    CUDA ops that the CPU torch.compile path can't handle. Built once per
    session so the compiled-model cache carries over between tests that
    reuse it (e.g. dynamic-shape coverage).

    Uses bfloat16, the project's recommended default -- this fixture is for
    tests that don't need byte-for-byte parity with eager (e.g. dynamic
    shapes). The exact-match correctness test builds its own float32
    instances instead; see test_run.py for why.
    """
    select_cpu_platform()
    from vllm import LLM

    llm = LLM(
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
    yield llm
    del llm
