# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import paddle

from save_model import exportModel

def meshgrid():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x, y, z):
        return paddle.meshgrid(x, y, z)

    x = paddle.randint(low=0, high=100, shape=[5])
    y = paddle.randint(low=0, high=100, shape=[3])
    z = paddle.randint(low=0, high=100, shape=[2])
    return exportModel('meshgrid', test_model, [x, y, z], target_dir=sys.argv[1])

if __name__ == "__main__":
    meshgrid()
