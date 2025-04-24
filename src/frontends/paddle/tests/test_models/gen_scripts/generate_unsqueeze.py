# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# unsqueeze paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'

def unsqueeze(name : str, x, axes : list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype = data_type)
        if paddle.__version__ >= '2.0.0':
            out = paddle.unsqueeze(node_x, axis=axes, name='unsqueeze')
        else:
            out = paddle.fluid.layers.unsqueeze(node_x, axes=axes, name='unsqueeze')

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
    data = np.random.rand(5, 10).astype(data_type)

    unsqueeze("unsqueeze", data, [1])

if __name__ == "__main__":
    main()
