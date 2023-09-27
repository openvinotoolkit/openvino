# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# tanh_shrink paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import paddle.nn.functional as F
import sys

data_type = "float32"


def tanh_shrink(name: str, x):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        out = F.tanhshrink(data)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feedkeys=["x"],
            fetchlist=[out],
            inputs=[x],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    x = np.random.uniform(-1000, 1000, (8, 24, 32)).astype(data_type)

    tanh_shrink("tanh_shrink", x)


if __name__ == "__main__":
    main()
