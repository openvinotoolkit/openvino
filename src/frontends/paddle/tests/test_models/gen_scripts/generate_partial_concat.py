# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# partial_concat paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = "float32"


def partial_concat(name: str, x, y, start_index=0, length=-1):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        X = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        Y = paddle.static.data(name="y", shape=x.shape, dtype=data_type)
        sum_out = paddle.fluid.contrib.layers.partial_concat(
            [X, Y], start_index=start_index, length=length
        )

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x, "y": y}, fetch_list=[sum_out])

        saveModel(
            name,
            exe,
            feedkeys=["x", "y"],
            fetchlist=[sum_out],
            inputs=[x, y],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    x = np.random.uniform(-100, 100, (2, 3)).astype(data_type)
    y = np.random.uniform(-100, 100, (2, 3)).astype(data_type)
    partial_concat("partial_concat", x, y)


if __name__ == "__main__":
    main()
