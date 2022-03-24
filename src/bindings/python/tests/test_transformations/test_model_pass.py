# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.runtime.passes import Manager

from utils.utils import get_test_function, MyModelPass


def test_model_pass():
    m = Manager()
    p = m.register_pass(MyModelPass())
    m.run_passes(get_test_function())

    assert p.model_changed
