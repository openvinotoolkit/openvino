# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# gelu paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'

def gelu(name:str, x, approximate=False):
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name='x', shape=x.shape, dtype = data_type)
        if paddle.__version__ >= '2.0.0':
            out = paddle.nn.functional.gelu(data, approximate=approximate)
        else:
            out = paddle.fluid.layers.gelu(data, approximate=approximate)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[data], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    x = np.random.rand(8, 24, 32).astype(data_type)

    gelu("gelu_erf", x)
    gelu("gelu_tanh", x, True)

if __name__ == "__main__":
    main()
