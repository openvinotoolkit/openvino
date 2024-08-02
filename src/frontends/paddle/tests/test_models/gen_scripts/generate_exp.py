# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# exp paddle model generator
#

import paddle
import numpy as np
from save_model import saveModel
import sys


def exp(name: str, x):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        if paddle.__version__ >= '2.0.0':
            out = paddle.exp(x=node_x)
        else:
            out = paddle.fluid.layers.exp(x=node_x)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[
                  x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    input_shape = (1, 2, 3)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    exp("exp_test_float32", input_data)


if __name__ == "__main__":
    main()
