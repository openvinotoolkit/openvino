# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# squeeze paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'

def squeeze(name : str, x, axes : list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype = data_type)
        if paddle.__version__ >= '2.0.0':
            out = paddle.squeeze(node_x, axis=axes, name='squeeze')
        else:
            out = paddle.fluid.layers.squeeze(node_x, axes=axes, name='squeeze')
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
    data = np.random.rand(1, 3, 1, 4).astype(data_type)

    squeeze("squeeze", data, [0, -2])
    squeeze("squeeze_null_axes", data, [])

if __name__ == "__main__":
    main()
