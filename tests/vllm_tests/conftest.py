# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the vLLM + OpenVINO torchdynamo backend test suite."""

import gc
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


@pytest.fixture(autouse=True)
def _gc_between_tests():
    """Force a collection after every test.

    A test's own `del llm` drops the last Python reference, but nothing
    guarantees the collector runs before the next test starts -- without
    this, a lingering reference can keep one test's memory resident into
    the next.
    """
    yield
    gc.collect()


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
    CPU torch.compile path can't handle. Every test builds and deletes its
    own instance: sharing one across tests kept it resident for as long as
    any test using it was still running, stacking its footprint onto every
    other test's own peak (a real OOM cause on CI, not just a style choice).
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
