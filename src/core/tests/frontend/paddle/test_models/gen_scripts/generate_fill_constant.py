# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# fill_const paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def fill_constant(name : str, shape : list, dtype, value):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x1 = paddle.fluid.layers.fill_constant(shape=shape, value=value, dtype=dtype, name='fill_constant')
        x2 = paddle.fluid.layers.fill_constant(shape=shape, value=value, dtype=dtype, name='fill_constant')
        out = paddle.add(paddle.cast(x1, np.float32), paddle.cast(x2, np.float32))
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=[], fetchlist=[out], inputs=[], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def fill_constant_tensor(name : str, shape : list, dtype, value):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_value = paddle.static.data(name='value', shape=[1], dtype=dtype)
        x1 = paddle.fluid.layers.fill_constant(shape=shape, value=node_value, dtype=dtype, name='fill_constant1')
        out = paddle.cast(x1, np.float32)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={"value": value},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=["value"], fetchlist=[out], inputs=[np.array([value]).astype(dtype)], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def fill_constant_shape_tensor(name : str, shape, dtype, value):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_shape = paddle.fluid.layers.fill_constant(shape=[2], value=shape, dtype='int32', name='shape')
        x1 = paddle.fluid.layers.fill_constant(shape=node_shape, value=value, dtype=dtype, name='fill_constant')
        out = paddle.cast(x1, np.float32)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            fetch_list=[out])

        saveModel(name, exe, feedkeys=[], fetchlist=[out], inputs=[], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def fill_constant_shape_tensor_list(name : str, shape: list, dtype, value):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_shape = paddle.fluid.layers.fill_constant(shape=[1], value=shape, dtype='int32', name='shape')
        x1 = paddle.fluid.layers.fill_constant(shape=[2, node_shape], value=value, dtype=dtype, name='fill_constant')
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
    fill_constant("fill_constant", [2, 3, 4], 'float32', 0.03)
    fill_constant("fill_constant_int32", [2, 3, 4], "int32", 2)
    fill_constant("fill_constant_int64", [2, 3, 4], "int64", 4)
    fill_constant_tensor("fill_constant_tensor", [2, 3, 4], 'float32', 0.05)
    fill_constant_shape_tensor("fill_constant_shape_tensor", 2, 'float32', 0.05)
    fill_constant_shape_tensor_list("fill_constant_shape_tensor_list", 2, 'float32', 0.05)


if __name__ == "__main__":
    main()