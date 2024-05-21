# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel
import sys
data_type = 'float32'


def paddle_argmax(name : str, x, axis):
    import paddle
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        out = paddle.argmax(x=node_x, axis=axis)
        out = paddle.cast(out, np.float32)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def paddle_argmax1(name : str, x):
    import paddle
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        out = paddle.argmax(x=node_x)
        out = paddle.cast(out, np.float32)
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
    data = np.random.random([3,5,7,2]).astype("float32")
    axis = 0
    paddle_argmax("argmax", data, axis)
    paddle_argmax1("argmax1", data)


if __name__ == "__main__":
    main()     
