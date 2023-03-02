# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# index_select paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def index_select(name: str, x, y, axis = 0):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        index = paddle.static.data(name='index', shape=y.shape, dtype=y.dtype)

        out = paddle.index_select(data, index, axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'index': y},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x', 'index'], fetchlist=[
                out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    # For multi-dimension input
    x_shape = (11, 20)
    x_type = "float32"
    index = [1, 3, 5]
    index_type = "int32"

    xnp = np.random.random(x_shape).astype(x_type)
    index_np = np.array(index).astype(index_type)

    index_select("index_select_multi_dimension", xnp, index_np)

    # For one_dimension input
    x_shape = (10000)
    x_type = "int64"
    index = [1, 3, 5]
    index_type = "int64"

    xnp = np.random.random(x_shape).astype(x_type)
    index_np = np.array(index).astype(index_type)

    index_select("index_select_one_dimension", xnp, index_np)

    # For axis as input
    x_shape = (100, 80, 3)
    x_type = "float32"
    index = [1, 3, 5]
    index_type = "int32"

    xnp = np.random.random(x_shape).astype(x_type)
    index_np = np.array(index).astype(index_type)

    index_select("index_select_with_axis", xnp, index_np, 1)


if __name__ == "__main__":
    main()
