# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# grid_sample paddle model generator
#
import paddle
import numpy as np
from save_model import saveModel
import sys


def grid_sample(name: str, x, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_node = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        grid_node = paddle.static.data(name="grid", shape=grid.shape, dtype=grid.dtype)
        out = paddle.nn.functional.grid_sample(x_node, grid_node, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(feed={"x": x, "grid": grid}, fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'grid'], fetchlist=[out], inputs=[x, grid], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    x_shape = [2, 2, 3, 3]
    grid_shape = [2, 3, 3, 2]
    dtype = np.float32
    mode = "bilinear"
    padding_mode = "zeros"
    align_corners = False
    x = np.random.randn(*(x_shape)).astype(dtype)
    grid = np.random.uniform(-1, 1, grid_shape).astype(dtype)
    grid_sample(name='grid_sample_1', x=x, grid=grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


if __name__ == "__main__":
    main()
