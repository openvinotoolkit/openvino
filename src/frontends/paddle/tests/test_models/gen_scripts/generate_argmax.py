# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel
import sys
data_type = 'float32'


def paddle_argmax(name : str, x, axis, keepdim=False):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        out = paddle.argmax(x=node_x, axis=axis, keepdim=keepdim)
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

def paddle_argmin(name : str, x, axis, keepdim=False):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        out = paddle.argmin(x=node_x, axis=axis, keepdim=keepdim)
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
    # issue 37827: keepdims=True must keep the reduced axis (shape [2,2,1] for axis=2)
    data2 = np.array([[[1.0, 5.0, 3.0], [2.0, 4.0, 9.0]],
                      [[3.0, 2.0, 1.0], [8.0, 7.0, 6.0]]]).astype(np.float32)
    paddle_argmax("argmax_keepdim", data2, axis=2, keepdim=True)
    paddle_argmin("argmin_keepdim", data2, axis=2, keepdim=True)
    # issue 37827: axis=None is exported as flatten=True; with keepdims=True the flattened
    # reduction keeps the rank of the input, all the dimensions being of size 1 (an (2,3,4)
    # input gives a (1,1,1) output). The extremes are moved away from the ends so that the
    # flattened index is not trivially the first or the last element.
    data3 = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    data3[0, 1, 2] = 5.0    # flattened index 6
    data3[1, 2, 1] = -5.0   # flattened index 21
    paddle_argmax("argmax_keepdim_flatten", data3, axis=None, keepdim=True)
    paddle_argmin("argmin_keepdim_flatten", data3, axis=None, keepdim=True)


if __name__ == "__main__":
    main()
