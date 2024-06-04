# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# tanh_shrink paddle model generator
#
import numpy as np
import sys
from save_model import saveModel


def tanh_shrink(name: str, x):
    import paddle

    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = paddle.nn.functional.tanhshrink(node_x)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={'x': x}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feed_vars=[node_x],
            fetchlist=[out],
            inputs=[x],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    data = np.random.uniform(10, 20, [2, 3, 4]).astype(np.float32)
    tanh_shrink("tanh_shrink_1", data)

    data = np.random.uniform(-10, 20, [4, 3, 2]).astype(np.float32)
    tanh_shrink("tanh_shrink_2", data)

if __name__ == "__main__":
    main()


