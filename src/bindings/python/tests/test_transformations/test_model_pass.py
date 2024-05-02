# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.runtime.passes import Manager
from tests.test_transformations.utils.utils import MyModelPass, get_relu_model


def test_model_pass():
    manager = Manager()
    model_pass = manager.register_pass(MyModelPass())
    manager.run_passes(get_relu_model())

    assert model_pass.model_changed
