# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.passes import Manager

from tests.test_transformations.utils.utils import get_relu_model, MyModelPass


def test_model_pass():
    manager = Manager()
    model_pass = manager.register_pass(MyModelPass())
    manager.run_passes(get_relu_model())

    assert model_pass.model_changed


def test_model_pass_run_on_model():
    model_pass = MyModelPass()
    model_pass.run_on_model(get_relu_model())

    assert model_pass.model_changed
