# -*- coding: utf-8 -*-
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.runtime.passes import Manager

from utils.utils import get_test_function, MyModelPass


def test_model_pass():
    manager = Manager()
    model_pass = manager.register_pass(MyModelPass())
    manager.run_passes(get_test_function())

    assert model_pass.model_changed
