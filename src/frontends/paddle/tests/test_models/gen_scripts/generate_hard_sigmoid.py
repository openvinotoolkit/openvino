# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# hard_sigmoid paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def hard_sigmoid(name: str, x, slope: float = 0.2, offset: float = 0.5, data_type='float32'):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype = data_type)
        out = paddle.nn.functional.hardsigmoid(node_x, slope=slope, offset=offset, name='hard_sigmoid')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data_type = 'float32'
    data = np.array([0, 1, 2, 3, 4, 5, 6, -10]).astype(data_type)

    hard_sigmoid("hard_sigmoid", data, 0.1, 0.6, data_type)


if __name__ == "__main__":
    main()
