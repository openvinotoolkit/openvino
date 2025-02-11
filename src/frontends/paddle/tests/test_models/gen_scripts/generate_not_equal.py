# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# not_equal paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def not_equal(name: str, x, y, data_type, cast_to_fp32=False):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=data_type)
        out = paddle.not_equal(node_x, node_y)
        # FuzzyTest framework doesn't support boolean so cast to fp32/int32
        if cast_to_fp32:
                data_type = "float32"

        out = paddle.cast(out, np.float32)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out],
                inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    test_cases = [
        "float32",
        "int32",
        "int64"
    ]

    for test in test_cases:
        data_x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(test)
        data_y = np.array([[[[2, 0, 3]], [[3, 1, 4]]]]).astype(test)

        if ((test == "float64") or (test == "int64")):
            not_equal("not_equal_" + test, data_x, data_y, test, True)
        else:
            not_equal("not_equal_" + test, data_x, data_y, test, False)


if __name__ == "__main__":
    main()
