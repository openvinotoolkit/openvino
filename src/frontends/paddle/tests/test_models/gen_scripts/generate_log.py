# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# log paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def log(name: str, x, data_type='float32'):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        if paddle.__version__ >= '2.0.0':
            out = paddle.log(node_x, name='log')
        else:
            out = paddle.fluid.layers.log(node_x, name='log')

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
    x = np.array([0, 1, 2, -10]).astype(data_type)
    log("log", x)


if __name__ == "__main__":
    main()
