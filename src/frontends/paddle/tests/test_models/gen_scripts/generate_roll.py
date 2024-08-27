# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# round paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'


def paddle_roll(name: str, x, shifts, axis=None):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_node = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        out = paddle.roll(x_node, shifts, axis) if axis is not None else paddle.roll(
            x_node, shifts)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[x_node], fetchlist=[out], inputs=[
                  x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    x = np.random.randn(2, 3, 4).astype(data_type)

    paddle_roll("roll_test_0", x, shifts=[1])
    paddle_roll("roll_test_1", x, shifts=[1], axis=[0])
    paddle_roll("roll_test_2", x, shifts=1, axis=0)
    paddle_roll("roll_test_3", x, shifts=[0, 1], axis=[0, 1])


if __name__ == "__main__":
    main()
