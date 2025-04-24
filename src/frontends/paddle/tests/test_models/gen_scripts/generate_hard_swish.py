# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# sigmoid paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def hard_swish(name: str, x, threshold=6.0, scale=6.0, offset=3.0, data_type='float32'):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        if paddle.__version__ >= '2.0.0':
            out = paddle.nn.functional.hardswish(node_x, name='hard_swish')
        else:
            out = paddle.fluid.layers.hard_swish(node_x, threshold=threshold, scale=scale, offset=offset, name='hard_swish')

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
    data = np.array([-6, 1, 6]).astype(data_type)
    hard_swish("hard_swish", data, data_type='float32')


if __name__ == "__main__":
    main()
