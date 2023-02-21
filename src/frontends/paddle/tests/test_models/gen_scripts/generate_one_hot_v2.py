# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# one_hot_v2 paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def one_hot_v2(name: str, x, num_classes):
    import paddle

    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        out = paddle.one_hot(x, num_classes=num_classes)
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        outs = exe.run(feed={"x": x}, fetch_list=[out])
        saveModel(
            name,
            exe,
            feedkeys=["x"],
            fetchlist=[out],
            inputs=[x],
            outputs=[out],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    test_cases = ["int32", "int64"]

    for test in test_cases:
        data_x = np.array([0, 1, 2, 3, 4, 5]).astype(test)
        one_hot_v2("one_hot_v2_" + test, data_x, 6)


if __name__ == "__main__":
    main()
