# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# fill_const paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def abs(name : str, x):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        if paddle.__version__ >= '2.0.0':
            abs_node = paddle.abs(node_x, name='abs_node')
        else:
            abs_node = paddle.fluid.layers.abs(node_x, name='abs_node')
        out = paddle.assign(abs_node)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             
        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    input_x = np.array([-0.4, -0.2, 0.1, 0.3]).astype(np.float32)
    abs("abs_float32", input_x)

if __name__ == "__main__":
    main()