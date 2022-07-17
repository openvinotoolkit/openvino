# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# elementwise_floordiv paddle model generator
#
import numpy as np
import sys
from save_model import saveModel

def elementwise_floordiv(name : str, x, y, axis, in_dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = paddle.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = paddle.fluid.layers.nn.elementwise_floordiv(node_x, node_y, axis=axis)

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
    elementwise_floordiv("elementwise_floordiv" + name, data_x, data_y, axis, in_dtype)

def main():

    in_dtype = 'int64'
    data_x = np.array([2, 3, 4]).astype(in_dtype)
    data_y = np.array([1, 5, 2]).astype(in_dtype)
    axis = -1
    elementwise_ops("1", data_x, data_y, axis, in_dtype)

    # data_y's shape is the continuous subsequence of data_x's shape
    data_x = np.random.randint(1, 5, size=[2, 3, 4, 5]).astype(in_dtype)
    data_y = np.random.randint(1, 5, size=[3, 4]).astype(in_dtype)
    axis = 1
    elementwise_ops("2", data_x, data_y, axis, in_dtype)

    data_y = np.random.randint(1, 5, size=[5]).astype(in_dtype)
    axis = 3
    elementwise_ops("3", data_x, data_y, axis, in_dtype)

if __name__ == "__main__":
    main()

