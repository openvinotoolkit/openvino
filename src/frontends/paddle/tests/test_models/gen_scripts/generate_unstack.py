# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# unstack paddle model generator
#
import paddle
import numpy as np
from save_model import saveModel
import sys


def unstack(name: str, x, axis):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_node = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        out = paddle.unstack(x_node, axis) if axis is not None else paddle.unstack(x_node)
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(feed={"x": x}, fetch_list=[out])
        saveModel(name, exe, feed_vars=[x_node], fetchlist=out, inputs=[x], outputs=outs, target_dir=sys.argv[1])

    return outs


def main():
    dtype = np.float32
    x = np.random.randn(2, 3, 4).astype(dtype)
    unstack(name='unstack_1', x=x, axis=0)

    dtype = np.int32
    x = np.random.randn(2, 3, 4).astype(dtype)
    unstack(name='unstack_2', x=x, axis=1)

    dtype = np.int64
    x = np.random.randn(3, 4).astype(dtype)
    unstack(name='unstack_3', x=x, axis=-1)
    unstack(name='unstack_4', x=x, axis=None)

    x = np.random.randn(2, 1, 4).astype(dtype)
    unstack(name='unstack_5', x=x, axis=0)

if __name__ == "__main__":
    main()
