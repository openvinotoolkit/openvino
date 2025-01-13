# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from save_model import saveModel
import sys


def paddle_assign_value(name, test_x):
    import paddle
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        node_x = paddle.static.data(name='x', shape=test_x.shape, dtype=test_x.dtype)
        node_x = paddle.cast(node_x, dtype=test_x.dtype)
        const_value = paddle.assign(test_x, output=None)
        result = paddle.cast(paddle.concat([node_x, const_value], 0), dtype=np.float32)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': test_x},
            fetch_list=[result]
        )

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[result], inputs=[test_x], outputs=[outs[0]], target_dir=sys.argv[1])


def compare():

    test_cases = [
        {
            "name": "assign_value_fp32",
            "input": np.ones([1, 1, 4, 4]).astype(np.float32)
        },
        {
            "name": "assign_value_int32",
            "input": np.ones([1, 1, 4, 4]).astype(np.int32)
        },
        {
            "name": "assign_value_int64",
            "input": np.ones([1, 1, 4, 4]).astype(np.int64)
        },
        {
            "name": "assign_value_boolean",
            "input": np.array([False, True, False])
        }
    ]
    for test in test_cases:
        paddle_assign_value(test['name'], test['input'])


if __name__ == "__main__":
    compare()
