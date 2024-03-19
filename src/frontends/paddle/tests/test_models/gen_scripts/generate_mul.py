# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from save_model import saveModel
import sys

def paddle_matmul(name, x1, x2, x_transpose=False, y_transpose=False):
    import paddle

    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x1 = paddle.static.data(name='x1', shape=x1.shape, dtype=x1.dtype)
        node_x2 = paddle.static.data(name='x2', shape=x2.shape, dtype=x2.dtype)
        if paddle.__version__ >= '2.0.0':
            mul_node = paddle.matmul(node_x1, node_x2, x_transpose, y_transpose)
        else:
            mul_node = paddle.fluid.layers.matmul(node_x1, node_x2, x_transpose, y_transpose)
        result = paddle.static.nn.batch_norm(mul_node, use_global_stats=True)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x1': x1, 'x2': x2},
            fetch_list=[result])
        saveModel(name, exe, feed_vars=[node_x1, node_x2], fetchlist=[result], inputs=[x1, x2], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


if __name__ == "__main__":
    input_2x5 = np.array([[1, 2, 3, 4, 5],
                       [6, 7, 8, 9, 10]]).astype(np.float32)

    input_5x3 = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12],
                       [13, 14, 15]]).astype(np.float32)

    input_5x2 = np.array([[1, 2],
                          [4, 5],
                          [7, 8],
                          [10, 11],
                          [13, 14]]).astype(np.float32)

    input_2x3 = np.array([[1, 2, 3],
                          [4, 5, 6]]).astype(np.float32)

    paddle_matmul("matmul_xt", input_2x5, input_2x3, x_transpose=True, y_transpose=False)
    paddle_matmul("matmul_yt", input_2x3, input_5x3, x_transpose=False, y_transpose=True)
    paddle_matmul("matmul_xt_yt", input_2x5, input_5x2, x_transpose=True, y_transpose=True)
