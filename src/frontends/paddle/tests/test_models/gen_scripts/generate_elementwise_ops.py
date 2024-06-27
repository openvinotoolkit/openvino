# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# elementwise paddle model generator
#
import numpy as np
import sys
from save_model import saveModel
import paddle

if paddle.__version__ >= '2.6.0':
    import paddle.base as fluid
else:
    import paddle.fluid as fluid

def elementwise_add(name: str, x, y, in_dtype, axis=-1):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        if paddle.__version__ >= '2.0.0':
            out = paddle.add(node_x, node_y)
        else:
            out = fluid.layers.elementwise_add(
                node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[
                  x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_sub(name: str, x, y, in_dtype, axis=-1):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        if paddle.__version__ >= '2.0.0':
            out = paddle.subtract(node_x, node_y)
        else:
            out = fluid.layers.elementwise_sub(
                node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[
                  x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_div(name: str, x, y, in_dtype, axis=-1):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        if paddle.__version__ >= '2.0.0':
            out = paddle.divide(node_x, node_y)
        else:
            out = fluid.layers.elementwise_div(
                node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[
                  x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_mod(name: str, x, y, in_dtype, is_api=False, axis=-1):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        if paddle.__version__ >= '2.0.0':
            out = paddle.floor_mod(node_x, node_y)
        else:
            if is_api:
                out = paddle.floor_mod(node_x, node_y)
            else:
                out = fluid.layers.elementwise_mod(
                    node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[
                  x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_mul(name: str, x, y, in_dtype, axis=-1):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        if paddle.__version__ >= '2.0.0':
            out = paddle.multiply(node_x, node_y)
        else:
            out = fluid.layers.elementwise_mul(
                node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[
                  x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_mul_bool(name: str, x, y, in_dtype='bool'):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        mul = node_x * node_y
        out = paddle.cast(mul, 'float32')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[
                  x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_min(name: str, x, y, in_dtype, axis=-1):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        if paddle.__version__ >= '2.0.0':
            out = paddle.minimum(node_x, node_y)
        else:
            out = fluid.layers.elementwise_min(
                node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[
                  x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_max(name: str, x, y, in_dtype, axis=-1):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        if paddle.__version__ >= '2.0.0':
            out = paddle.maximum(node_x, node_y)
        else:
            out = fluid.layers.elementwise_max(
                node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[
                  x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_pow(name: str, x, y, in_dtype, axis=-1):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        if paddle.__version__ >= '2.0.0':
            out = paddle.pow(node_x, node_y)
        else:
            out = fluid.layers.elementwise_pow(
                node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[
                  x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_floordiv(name: str, x, y, in_dtype, axis=-1):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=in_dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=in_dtype)
        if paddle.__version__ >= '2.0.0':
            out = paddle.floor_divide(node_x, node_y)
        else:
            out = fluid.layers.nn.elementwise_floordiv(
                node_x, node_y, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[
                  x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def elementwise_ops(name: str, data_x, data_y, in_dtype, axis=-1):
    elementwise_add("elementwise_add" + name, data_x, data_y, in_dtype, axis)
    elementwise_sub("elementwise_sub" + name, data_x, data_y, in_dtype, axis)
    elementwise_div("elementwise_div" + name, data_x, data_y, in_dtype, axis)
    elementwise_mod("elementwise_mod" + name, data_x, data_y, in_dtype, axis)
    elementwise_mul("elementwise_mul" + name, data_x, data_y, in_dtype, axis)
    elementwise_min("elementwise_min" + name, data_x, data_y, in_dtype, axis)
    elementwise_max("elementwise_max" + name, data_x, data_y, in_dtype, axis)
    elementwise_pow("elementwise_pow" + name, data_x, data_y, in_dtype, axis)


def main():

    in_dtype = 'float32'
    data_x = np.array([2, 3, 4]).astype(in_dtype)
    data_y = np.array([1, 5, 2]).astype(in_dtype)
    elementwise_ops("1", data_x, data_y, in_dtype)
    elementwise_mod('floor_mod1', data_x, data_y, in_dtype, True)

    # data_y's shape is the continuous subsequence of data_x's shape
    data_x = np.random.rand(2, 5, 3, 4).astype(np.float32)
    data_y = (0.1 + np.random.rand(3, 4).astype(np.float32)) / 1.1
    elementwise_ops("2", data_x, data_y, in_dtype)
    elementwise_mod('floor_mod2', data_x, data_y, in_dtype, True)

    data_y = (0.1 + np.random.rand(4).astype(np.float32)) / 1.1

    if paddle.__version__ >= '2.0.0':
        elementwise_ops("3", data_x, data_y, in_dtype)
    else:
        elementwise_ops("3", data_x, data_y, in_dtype, 1)

    data_y = (0.1 + np.random.rand(5, 3, 4).astype(np.float32)) / 1.1
    if paddle.__version__ >= '2.0.0':
        elementwise_ops("4", data_x, data_y, in_dtype)
    else:
        elementwise_ops("4", data_x, data_y, in_dtype, 0)

    # test for elementwise_floordiv, support int and int64
    # paddle1.8 support axis = [0, x_last_dims]
    # paddle2.x only support axis = -1
    floordiv_support_dtype = ['int64', 'int32']
    data_x = np.array([-4, 0, -8])

    data_y = np.array([3, 5, 3])
    for dtype in floordiv_support_dtype:
        elementwise_floordiv("elementwise_floordiv_" + dtype + "_1",
                             data_x.astype(dtype), data_y.astype(dtype), dtype)

    data_x = np.random.randint(-10, 10, [2, 5, 3, 4])
    data_y = np.random.randint(1, 5, [3, 4])
    for dtype in floordiv_support_dtype:
        elementwise_floordiv("elementwise_floordiv_" + dtype + "_2",
                             data_x.astype(dtype), data_y.astype(dtype), dtype)

    data_y = np.random.randint(1, 5, [5, 3, 4])
    for dtype in floordiv_support_dtype:
        elementwise_floordiv("elementwise_floordiv_" + dtype + "_3",
                             data_x.astype(dtype), data_y.astype(dtype), dtype)

    # test for elementwise_mul with bool data type
    sample_arr = [True, False]
    data_x = np.random.choice(sample_arr, size=(2, 3, 4))
    data_y = np.random.choice(sample_arr, size=(1, 3, 4))
    elementwise_mul_bool("elementwise_mul_bool1", data_x, data_y)


if __name__ == "__main__":
    main()
