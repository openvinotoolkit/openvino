# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino import Tensor, Type


def test_run():
    tensor = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
    assert tensor.element_type == Type.f32


if __name__ == "__main__":
    test_run()
