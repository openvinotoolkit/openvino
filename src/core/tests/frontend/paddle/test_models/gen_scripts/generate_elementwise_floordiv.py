# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# elementwise_floordiv paddle model generator
#
import numpy as np
import sys
from save_model import saveModel

def elementwise_floordiv(name : str, x, y, in_dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        node_y = paddle.static.data(name = 'y', shape = y.shape, dtype = in_dtype)
        out = paddle.floor_divide(node_x, node_y)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    for in_dtype in ['int64', 'int32']:
        data_x = np.array([2, 3, 4]).astype(in_dtype)
        data_y = np.array([1, 5, 2]).astype(in_dtype)
        elementwise_floordiv(f"elementwise_floordiv1_{in_dtype}", data_x, data_y, in_dtype)

        # data_y's shape is the continuous subsequence of data_x's shape
        data_x = np.random.randint(1, 5, size=[2, 3, 4, 5]).astype(in_dtype)
        data_y = np.random.randint(-10, -5, size=[2, 3, 4, 5]).astype(in_dtype)
        elementwise_floordiv(f"elementwise_floordiv2_{in_dtype}", data_x, data_y, in_dtype)

        data_y = np.random.randint(1, 5, size=[5]).astype(in_dtype)
        elementwise_floordiv(f"elementwise_floordiv3_{in_dtype}", data_x, data_y, in_dtype)

if __name__ == "__main__":
    main()

