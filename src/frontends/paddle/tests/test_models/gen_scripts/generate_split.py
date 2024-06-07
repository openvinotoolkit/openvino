# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# split paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def split(name : str, x, attrs : dict):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        
        if paddle.__version__ >= '2.0.0':
            out = paddle.split(node_x, num_or_sections=attrs['num_or_sections'], axis=attrs['axis'])
        else:
            out = paddle.fluid.layers.split(node_x, num_or_sections=attrs['num_or_sections'], dim=attrs['axis'])
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])
        print("outputs: ", type(outs),len(outs))
        print("out: ", type(out), len(out))

        saveModel(name, exe, feed_vars=[node_x], fetchlist=out, inputs=[x], outputs=outs, target_dir=sys.argv[1])

    return outs[0]


def split_dim_tensor(name : str, x, attrs : dict, dim):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        dim_node = paddle.assign(dim)
        if paddle.__version__ >= '2.0.0':
            out = paddle.split(node_x, num_or_sections=attrs['num_or_sections'], axis=dim_node)
        else:
            out = paddle.fluid.layers.split(node_x, num_or_sections=attrs['num_or_sections'], dim=dim_node)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])
        print("outputs: ", type(outs),len(outs))
        print("out: ", type(out), len(out))

        saveModel(name, exe, feed_vars=[node_x], fetchlist=out, inputs=[x], outputs=outs, target_dir=sys.argv[1])

    return outs[0]


def split_test_list_tensor(name : str, x, attrs : dict):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        section = attrs['num_or_sections']
        section[0] = paddle.assign(np.array((section[0],)).astype('int32'))
        if paddle.__version__ >= '2.0.0':
            out = paddle.split(node_x, num_or_sections=section, axis=attrs['axis'])
        else:
            out = paddle.fluid.layers.split(node_x, num_or_sections=section, dim=attrs['axis'])
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])
        print("outputs: ", type(outs),len(outs))
        print("out: ", type(out), len(out))

        saveModel(name, exe, feed_vars=[node_x], fetchlist=out, inputs=[x], outputs=outs, target_dir=sys.argv[1])

    return outs[0]


def main():
    # split
    data_types = ['float32'] #TODOD: ['bool', 'float16', 'float32', 'float64', 'int32', 'int64']
    num_or_sections = [3, [2, 3, 4], [2, 3, -1]]
    axes = [1, -2]

    idx = 1
    for t in data_types:
        for s in num_or_sections:
            for i in axes:
                paddle_attrs = {
                    'num_or_sections': s,
                    'axis': i
                }
                data_NCHW = np.random.rand(3,9,5).astype(t)
                split("split_test{}".format(idx), data_NCHW, paddle_attrs)
                idx+=1

    split("split_test_list", data_NCHW, {
        'num_or_sections': [4, 5],
        'axis': 1})
    split_dim_tensor("split_test_dim_int32", data_NCHW, {
        'num_or_sections': 3}, np.array([1,]).astype('int32'))
    split_dim_tensor("split_test_dim_int64", data_NCHW, {
        'num_or_sections': 3}, np.array([1,]).astype('int64'))
    split_test_list_tensor("split_test_list_tensor", data_NCHW, {
        'num_or_sections': [4, 5],
        'axis': 1})


if __name__ == "__main__":
    main()
