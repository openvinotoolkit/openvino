# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# reshape paddle model generator
#
import numpy as np
from save_model import saveModel
import sys

data_type = 'float32'


def reshape(name : str, x, out_shape):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        if paddle.__version__ >= '2.0.0':
            out = paddle.reshape(x=node_x, shape=out_shape)
        else:
            out = paddle.fluid.layers.reshape(x=node_x, shape=out_shape)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def reshape_tensor(name : str, x, out_shape, use_tensor_in_list):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        if use_tensor_in_list:
            out_shape[0] = paddle.assign(np.array((out_shape[0],)).astype('int32'))
            if paddle.__version__ >= '2.0.0':
                out = paddle.reshape(x=node_x, shape=out_shape)
            else:
                out = paddle.fluid.layers.reshape(x=node_x, shape=out_shape)
        else:
            out_shape = np.array(out_shape).astype('int32')
            node_shape = paddle.assign(out_shape)
            out = paddle.reshape(x=node_x, shape=node_shape)

        out = paddle.pow(out, 1)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)
    out_shape = [1, 1, 2, 8]
    reshape("reshape", data, out_shape)
    reshape_tensor("reshape_tensor", data, out_shape, False)
    reshape_tensor("reshape_tensor_list", data, out_shape, True)


if __name__ == "__main__":
    main()
