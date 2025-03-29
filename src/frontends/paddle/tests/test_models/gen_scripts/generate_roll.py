# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# round paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'


def paddle_roll(name: str, x, shifts, axis=None, shifts_is_var=False):
    paddle.enable_static()
    
    shifts = np.array([shifts], dtype='int64') if shifts_is_var else shifts

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_node = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        shifts_node = paddle.static.data(name="shifts", shape=[1], dtype='int64') if shifts_is_var else shifts
        out = paddle.roll(x_node, shifts_node, axis) if axis is not None else paddle.roll(
            x_node, shifts)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        
        feed_list = {'x': x, 'shifts': shifts} if shifts_is_var else {'x': x}
        outs = exe.run(
            feed=feed_list,
            fetch_list=[out])

        feed_vars = [x_node, shifts_node] if shifts_is_var else [x_node]
        input_list = [x, shifts] if shifts_is_var else [x]
        saveModel(name, exe, feed_vars=feed_vars, fetchlist=[out], inputs=input_list, outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    x = np.random.randn(2, 3, 4).astype(data_type)

    paddle_roll("roll_test_0", x, shifts=[1])
    paddle_roll("roll_test_1", x, shifts=[1], axis=[0])
    paddle_roll("roll_test_2", x, shifts=1, axis=0)
    paddle_roll("roll_test_3", x, shifts=[0, 1], axis=[0, 1])
    paddle_roll("roll_test_4", x, shifts=1, axis=0, shifts_is_var=True)


if __name__ == "__main__":
    main()
