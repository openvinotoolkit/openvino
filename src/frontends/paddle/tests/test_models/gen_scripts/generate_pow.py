# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pow paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def paddle_pow(name : str, x, y, data_type):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        if paddle.__version__ >= '2.0.0':
            y = paddle.to_tensor(y, dtype=data_type)
            out = paddle.pow(node_x, y, name='pow')
        else:
            out = paddle.fluid.layers.pow(node_x, y, name='pow')
        #FuzzyTest supports int32 & float32
        if data_type == "int64":
            out = paddle.cast(out, "float32")
        out = paddle.cast(out, "float32")
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def paddle_pow_tensor(name : str, x, y, data_type):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=data_type)
        if paddle.__version__ >= '2.0.0':
            out = paddle.pow(node_x, node_y, name='pow')
        else:
            out = paddle.fluid.layers.pow(node_x, node_y, name='pow')
        out = paddle.cast(out, "float32")

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out],
                  inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    test_cases = [
        {
            'name': "float32",
            'x': np.array([0, 1, 2, -10]).astype("float32"),
            'y': np.array([1.5]).astype("float32"),
            'dtype': "float32",
         },
        {
            'name': "int32",
            'x': np.array([0, 1, 2, -10]).astype("int32"),
            'y': np.array([2.0]).astype("float32"),
            'dtype': "int32"
        },
        {
            'name': "int64",
            'x': np.array([0, 1, 2]).astype("int64"),
            'y': np.array([30.0]).astype("float32"),
            'dtype': "int64"
        },
        {
            'name': "int64_out_of_range",
            'x': np.array([0, 1, 2]).astype("int64"),
            'y': np.array([40]).astype("float32"),
            'dtype': "int64"
        }
    ]

    for test in test_cases:
        paddle_pow("pow_" + test['name'], test['x'], test['y'], test['dtype'])

    x = np.array([0, 1, 2, -10]).astype("float32")
    y = np.array([2.0]).astype("float32")
    paddle_pow_tensor("pow_y_tensor", x, y, 'float32')


if __name__ == "__main__":
    main()
