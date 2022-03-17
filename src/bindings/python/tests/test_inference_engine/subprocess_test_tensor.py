# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.runtime import Tensor, Type


def test_run():
    t = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
    assert t.element_type == Type.f32


if __name__ == "__main__":
    test_run()
