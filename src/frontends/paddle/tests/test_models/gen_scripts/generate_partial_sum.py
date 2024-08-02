# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# partial_sum paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def partial_sum(name: str, x, y, start_index=0, length=-1):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_data = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        y_data = paddle.static.data(name="y", shape=x.shape, dtype=y.dtype)

        if paddle.__version__ >= '2.5.1':
            out = paddle.incubate.layers.nn.partial_sum(
                [x_data, y_data], start_index=start_index, length=length
            )
        else:
            out = paddle.fluid.contrib.layers.partial_sum(
            [x_data, y_data], start_index=start_index, length=length
            )


        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x, "y": y}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feed_vars=[x_data, y_data],
            fetchlist=[out],
            inputs=[x, y],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    dtype = 'float32'
    x = np.random.randn(6, 4).astype(dtype)
    y = np.random.randn(6, 4).astype(dtype)
    partial_sum("partial_sum_1", x, y, start_index=2, length=2)


    dtype = 'int32'
    x = np.random.randint(-10, 10, [5, 3]).astype(dtype)
    y = np.random.randint(-10, 10, [5, 3]).astype(dtype)
    partial_sum("partial_sum_2", x, y, start_index=1, length=-1)

    dtype = 'int64'
    x = np.random.randint(-10, 10, [8, 10]).astype(dtype)
    y = np.random.randint(-10, 10, [8, 10]).astype(dtype)
    partial_sum("partial_sum_3", x, y, start_index=1, length=5)

if __name__ == "__main__":
    main()
