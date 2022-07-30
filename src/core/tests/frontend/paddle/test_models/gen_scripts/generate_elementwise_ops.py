# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# elementwise paddle model generator
#
import numpy as np
import sys
from save_model import saveModel


def elementwise_add(name : str, x, y, axis, in_dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        out = paddle.fluid.layers.nn.elementwise_add(node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_sub(name : str, x, y, axis, in_dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        out = paddle.fluid.layers.nn.elementwise_sub(node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_div(name : str, x, y, axis, in_dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = paddle.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = paddle.fluid.layers.nn.elementwise_div(node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_mod(name : str, x, y, axis, in_dtype, is_api=False):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        if is_api:
            out = paddle.floor_mod(node_x, node_y)
        else:
            out = paddle.fluid.layers.elementwise_mod(node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_mul(name : str, x, y, axis, in_dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = paddle.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = paddle.fluid.layers.nn.elementwise_mul(node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_min(name : str, x, y, axis, in_dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = paddle.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = paddle.fluid.layers.nn.elementwise_min(node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_max(name : str, x, y, axis, in_dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = paddle.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = paddle.fluid.layers.nn.elementwise_max(node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_pow(name : str, x, y, axis, in_dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = paddle.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = paddle.fluid.layers.nn.elementwise_pow(node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def elementwise_ops(name : str, data_x, data_y, axis, in_dtype):
    elementwise_add("elementwise_add" + name, data_x, data_y, axis, in_dtype)
    elementwise_sub("elementwise_sub" + name, data_x, data_y, axis, in_dtype)
    elementwise_div("elementwise_div" + name, data_x, data_y, axis, in_dtype)
    elementwise_mod("elementwise_mod" + name, data_x, data_y, axis, in_dtype)
    elementwise_mul("elementwise_mul" + name, data_x, data_y, axis, in_dtype)
    elementwise_min("elementwise_min" + name, data_x, data_y, axis, in_dtype)
    elementwise_max("elementwise_max" + name, data_x, data_y, axis, in_dtype)
    elementwise_pow("elementwise_pow" + name, data_x, data_y, axis, in_dtype)


def main():

    in_dtype = 'float32'
    data_x = np.array([2, 3, 4]).astype(in_dtype)
    data_y = np.array([1, 5, 2]).astype(in_dtype)
    axis = -1
    elementwise_ops("1", data_x, data_y, axis, in_dtype)
    elementwise_mod('floor_mod1', data_x, data_y, -1, in_dtype, True)

    # data_y's shape is the continuous subsequence of data_x's shape
    data_x = np.random.rand(2, 5, 3, 4).astype(np.float32)
    data_y = (0.1 + np.random.rand(3, 4).astype(np.float32)) / 1.1
    elementwise_ops("2", data_x, data_y, axis, in_dtype)
    elementwise_mod('floor_mod2', data_x, data_y, -1, in_dtype, True)

    data_y = (0.1 + np.random.rand(5).astype(np.float32)) / 1.1
    axis = 1
    elementwise_ops("3", data_x, data_y, axis, in_dtype)

    data_y = (0.1 + np.random.rand(2, 5, 3).astype(np.float32)) / 1.1
    axis = 0
    elementwise_ops("4", data_x, data_y, axis, in_dtype)

if __name__ == "__main__":
    main()
