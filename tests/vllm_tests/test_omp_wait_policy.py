# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OV workers are spawned with OMP_WAIT_POLICY=PASSIVE; nothing else is.

Exercises the patched OMPProcessManager without loading a model. The
end-to-end check (the variable as seen inside real eager and OV workers) is
in test_run.py.
"""

import os
from types import SimpleNamespace

import pytest

from openvino.frontend.pytorch.torchdynamo.vllm import plugin


def _config(backend):
    return SimpleNamespace(compilation_config=SimpleNamespace(
        mode=SimpleNamespace(name="STOCK_TORCH_COMPILE"), backend=backend))


@pytest.fixture
def manager_cls(monkeypatch):
    """OMPProcessManager with the plugin patch applied and no inherited OMP_WAIT_POLICY."""
    import vllm.utils.ompmultiprocessing as omp_mp

    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    plugin._patch_omp_wait_policy()
    # A non-CPU platform makes the original __init__ return before it reads
    # the parallel config, and configure_omp_envs yield without touching env.
    monkeypatch.setattr(omp_mp, "current_platform", SimpleNamespace(is_cpu=lambda: False))
    return omp_mp.OMPProcessManager


@pytest.mark.precommit
@pytest.mark.parametrize("backend, expected", [("openvino", "PASSIVE"), ("inductor", None)])
def test_spawn_env_depends_on_backend(manager_cls, backend, expected):
    manager = manager_cls(_config(backend))
    with manager.configure_omp_envs(rank=0, local_rank=0):
        assert os.environ.get("OMP_WAIT_POLICY") == expected
    assert "OMP_WAIT_POLICY" not in os.environ


@pytest.mark.precommit
def test_user_value_wins(manager_cls, monkeypatch):
    monkeypatch.setenv("OMP_WAIT_POLICY", "ACTIVE")
    manager = manager_cls(_config("openvino"))
    with manager.configure_omp_envs(rank=0, local_rank=0):
        assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"
    assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"


@pytest.mark.precommit
def test_patch_is_idempotent(manager_cls):
    patched = manager_cls.configure_omp_envs
    plugin._patch_omp_wait_policy()
    assert manager_cls.configure_omp_envs is patched
