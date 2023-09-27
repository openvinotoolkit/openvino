# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# fill_const paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def full(name : str, shape : list, dtype, value):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x1 = paddle.full(shape=shape, fill_value=value, dtype=dtype, name='fill')
        x2 = paddle.full(shape=shape, fill_value=value, dtype=dtype, name='fill')
        out = paddle.add(paddle.cast(x1, np.float32), paddle.cast(x2, np.float32))
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=[], fetchlist=[out], inputs=[], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def full_tensor(name : str, shape : list, dtype, value):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_value = paddle.static.data(name='value', shape=[], dtype=dtype)
        x1 = paddle.full(shape=shape, fill_value=node_value, dtype=dtype, name='full1')
        out = paddle.cast(x1, np.float32)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={"value": value},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=["value"], fetchlist=[out], inputs=[np.array(value).astype(dtype)], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def full_shape_tensor(name : str, shape, dtype, value):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_shape = paddle.full(shape=[2], fill_value=shape, dtype='int32', name='shape')
        x1 = paddle.full(shape=node_shape, fill_value=value, dtype=dtype, name='full')
        out = paddle.cast(x1, np.float32)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            fetch_list=[out])

        saveModel(name, exe, feedkeys=[], fetchlist=[out], inputs=[], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def full_shape_tensor_list(name : str, shape: list, dtype, value):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_shape = paddle.full(shape=[1], fill_value=shape, dtype='int32', name='shape')
        x1 = paddle.full(shape=[2, node_shape], fill_value=value, dtype=dtype, name='full')
        out = paddle.cast(x1, np.float32)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            fetch_list=[out])

        saveModel(name, exe, feedkeys=[], fetchlist=[out], inputs=[], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    full("full", [2, 3, 4], 'float32', 0.03)
    full("full_int32", [2, 3, 4], "int32", 2)
    full("full_int64", [2, 3, 4], "int64", 4)
    full_tensor("full_tensor", [2, 3, 4], 'float32', 0.05)
    full_shape_tensor("full_shape_tensor", 2, 'float32', 0.05)
    full_shape_tensor_list("full_shape_tensor_list", 2, 'float32', 0.05)


if __name__ == "__main__":
    main()
