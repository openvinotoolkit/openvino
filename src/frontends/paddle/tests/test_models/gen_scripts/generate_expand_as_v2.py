# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# expand_as_v2 paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'


def expand_as_v2(name:str, x, y):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=data_type)
        out = paddle.expand_as(node_x, node_y, name='expand_as_v2')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out],
                  inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    data_x = np.random.rand(1, 1, 6).astype(data_type)
    data_y = np.random.rand(2, 3, 6).astype(data_type)
    expand_as_v2("expand_as_v2", data_x, data_y)

if __name__ == "__main__":
    main()
