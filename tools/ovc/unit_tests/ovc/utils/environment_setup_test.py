# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest

from openvino.frontend.tensorflow.utils import get_environment_setup as fe_get_environment_setup
from openvino.tools.ovc.environment_setup_utils import get_environment_setup as ovc_get_environment_setup


@pytest.fixture
def fake_tensorflow(monkeypatch):
    # Stands in for an installed TensorFlow: only the version attribute is read.
    module = types.ModuleType("tensorflow")
    module.__version__ = "2.20.0"
    monkeypatch.setitem(sys.modules, "tensorflow", module)
    return module


@pytest.mark.parametrize("get_environment_setup", [fe_get_environment_setup, ovc_get_environment_setup])
def test_environment_setup_reports_tensorflow_version(get_environment_setup, fake_tensorflow):
    env_setup = get_environment_setup("tf")
    assert env_setup["tensorflow"] == "2.20.0"
    assert env_setup["sys_platform"] == sys.platform


@pytest.mark.parametrize("get_environment_setup", [fe_get_environment_setup, ovc_get_environment_setup])
def test_environment_setup_without_tensorflow(get_environment_setup, monkeypatch):
    # a None entry in sys.modules makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    env_setup = get_environment_setup("tf")
    assert "tensorflow" not in env_setup
    assert env_setup["sys_platform"] == sys.platform
