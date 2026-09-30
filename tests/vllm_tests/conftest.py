# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the vLLM + OpenVINO torchdynamo backend test suite."""

import os

import pytest

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# test_run.py ships a plain function through collective_rpc; msgspec can't
# serialize callables, and this is local test-only IPC, not a network RPC.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

# Must precede the first `import vllm` (including importorskip below): vLLM
# reads this at module import, well before the plugin entry point loads.
# See vllm.preset.set_pre_import_env.
from openvino.frontend.pytorch.torchdynamo.vllm.preset import (  # noqa: E402
    set_pre_import_env,
)

set_pre_import_env()

pytest.importorskip("vllm")

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


def new_openvino_llm(**overrides):
    """Build a fresh OV-backend TinyLlama LLM instance (bfloat16).

    block_size=32 is a hard OV CPU PagedAttention kernel constraint;
    custom_ops=["none"] keeps vLLM from expanding RMSNorm/SiLU into ops the
    CPU torch.compile path can't handle. Each distinct call-shape compiles
    its own resident weight copy, so a test needing a shape the shared
    `openvino_llm` fixture doesn't already have should build its own
    instance here and delete it, rather than growing that fixture forever.
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


@pytest.fixture(scope="session")
def openvino_llm():
    """A single OV-backend TinyLlama LLM instance, shared across the session.

    Only for tests that reuse the *same* call shape as each other -- see
    new_openvino_llm for why a different shape needs its own instance.
    The exact-match correctness test builds its own float32 instances
    instead; see test_run.py for why.
    """
    llm = new_openvino_llm()
    yield llm
    del llm
