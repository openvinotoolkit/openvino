# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

#
# set_value paddle model generator
#
import numpy as np
import paddle
from save_model import saveModel


def paddle_set_value(name: str, x, value, callback, dtype):

    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=dtype)
        value_shape = (0,) if isinstance(value, (int, float)) else value.shape
        node_v = paddle.static.data(name="v", shape=value_shape, dtype=dtype)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        out = callback(paddle.clone(node_x), node_v)

        outs = exe.run(feed={"x": x, "v": value}, fetch_list=[out])

        saveModel(name, exe, feedkeys=['x', 'v'], fetchlist=[out], inputs=[x, value], outputs=[outs[0]], target_dir=sys.argv[1])
        

def main():
    shape = (2, 3, 4, 5)
    dtype = "float32"
    data = np.random.random(shape).astype(dtype)
    value = np.array([0]).astype(dtype)

    def set_value1(x, value):
        x[1:2, :, 2:4] = value
        return x

    paddle_set_value("set_value1", data, value, set_value1, dtype)

    shape = (5, 3, 1)
    dtype = "float32"
    data = np.random.random(shape).astype("float32")
    value = np.random.random((3, 3, 1)).astype(dtype)

    def set_value2(x, value):
        x[2:5, :] = value
        return x


    paddle_set_value("set_value2", data, value, set_value2, dtype)

    shape = (10, 2, 5)
    dtype = "int32"
    data = np.random.randint(0, 5, shape).astype(dtype)
    value = np.random.randint(0, 2, (3, 1, 1)).astype(dtype)
    def set_value3(x, value):
        x[-4:-1] = value
        return x

    paddle_set_value("set_value3", data, value, set_value3, dtype)

if __name__ == "__main__":
    main()