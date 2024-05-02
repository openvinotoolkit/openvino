# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import numpy as np
from save_model import saveModel


def paddle_bmm(x1, x2):
    import paddle

    paddle.enable_static()
    node_x1 = paddle.static.data(name="x1", shape=x1.shape, dtype=x1.dtype)
    node_x2 = paddle.static.data(name="x2", shape=x2.shape, dtype=x2.dtype)
    bmm_node = paddle.bmm(node_x1, node_x2)
    result = paddle.static.nn.batch_norm(bmm_node, use_global_stats=True)

    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(paddle.static.default_startup_program())

    outs = exe.run(feed={"x1": x1, "x2": x2}, fetch_list=[result])
    saveModel(
        "bmm",
        exe,
        feedkeys=[node_x1, node_x2],
        fetchlist=[result],
        inputs=[x1, x2],
        outputs=[outs[0]],
        target_dir=sys.argv[1],
        use_static_api=True,
    )

    return outs[0]


if __name__ == "__main__":
    input1 = np.array(
        [
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 7, 5) input tensor
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0, 29.0],
                [
                    30.0,
                    31.0,
                    32.0,
                    33.0,
                    34.0,
                ],
            ]
        ]
    ).astype(np.float32)

    input2 = np.ones([1, 5, 7]).astype("float32")
    paddle_result = paddle_bmm(input1, input2)
