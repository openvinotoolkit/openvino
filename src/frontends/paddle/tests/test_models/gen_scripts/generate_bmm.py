# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from save_model import saveModel
import sys


def paddle_bmm(x1, x2):
    import paddle

    paddle.enable_static()
    node_x1 = paddle.static.data(name='x1', shape=x1.shape, dtype=x1.dtype)
    node_x2 = paddle.static.data(name='x2', shape=x2.shape, dtype=x2.dtype)
    bmm_node = paddle.bmm(node_x1, node_x2)
    result = paddle.static.nn.batch_norm(bmm_node, use_global_stats=True)

    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(paddle.static.default_startup_program())

    outs = exe.run(
        feed={'x1': x1, 'x2': x2},
        fetch_list=[result])
    saveModel("bmm", exe, feed_vars=[node_x1, node_x2], fetchlist=[result],
              inputs=[x1, x2], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


if __name__ == "__main__":
    input1 = np.array([[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                        [5., 6., 7., 8., 9.],
                        [10., 11., 12., 13., 14.],
                        [15., 16., 17., 18., 19.],
                        [20., 21., 22., 23., 24.],
                        [25., 26., 27., 28., 29.],
                        [30., 31., 32., 33., 34.,]]]).astype(np.float32)

    input2 = np.ones([1, 5, 7]).astype('float32')
    paddle_result = paddle_bmm(input1, input2)
