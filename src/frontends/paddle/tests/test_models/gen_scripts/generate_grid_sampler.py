# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# grid_sampler paddle model generator
#
import paddle
import numpy as np
from save_model import saveModel
import sys


def grid_sampler(name: str, x, grid, mode="bilinear", padding_mode="zeros", align_corners=True, not_empty=True,
                 is_dynamic=False):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        if is_dynamic:
            x_node = paddle.static.data(name="x", shape=(-1, -1, -1, -1), dtype=x.dtype)
            grid_node = paddle.static.data(name="grid", shape=(-1, -1, -1, 2), dtype=grid.dtype)
        else:
            x_node = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
            grid_node = paddle.static.data(name="grid", shape=grid.shape, dtype=grid.dtype)
        out = paddle.nn.functional.grid_sample(x_node, grid_node, mode=mode, padding_mode=padding_mode,
                                               align_corners=align_corners) if not_empty else paddle.nn.functional.grid_sample(
            x_node, grid_node)
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(feed={"x": x, "grid": grid}, fetch_list=[out])
        saveModel(name, exe, feed_vars=[x_node, grid_node], fetchlist=[out], inputs=[x, grid], outputs=[outs[0]],
                  target_dir=sys.argv[1])

    return outs[0]


def main():
    dtype = np.float32
    x = np.random.randn(2, 2, 3, 3).astype(dtype)
    grid = np.random.uniform(-1, 1, [2, 3, 3, 2]).astype(dtype)
    grid_sampler(name='grid_sampler_1', x=x, grid=grid, not_empty=False)

    x = np.random.randn(2, 3, 5, 6).astype(dtype)
    grid = np.random.uniform(-1, 1, [2, 8, 9, 2]).astype(dtype)
    mode = "nearest"
    padding_mode = "reflection"
    align_corners = False
    grid_sampler(name='grid_sampler_2', x=x, grid=grid, mode=mode, padding_mode=padding_mode,
                 align_corners=align_corners)

    x = np.random.randn(2, 3, 128, 128).astype(dtype)
    grid = np.random.uniform(-1, 1, [2, 130, 130, 2]).astype(dtype)
    padding_mode = "border"
    grid_sampler(name='grid_sampler_3', x=x, grid=grid, mode=mode, padding_mode=padding_mode,
                 align_corners=align_corners)
    grid_sampler(name='grid_sampler_dyn', x=x, grid=grid, mode=mode, padding_mode=padding_mode,
                 align_corners=align_corners, is_dynamic=True)


if __name__ == "__main__":
    main()
