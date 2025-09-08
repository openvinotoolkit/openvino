# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# logical_or paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def equal_logical_or(name : str, x, y, z):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        node_y = paddle.static.data(name='y', shape=y.shape, dtype='float32')
        node_z = paddle.static.data(name='z', shape=z.shape, dtype='float32')

        bool_x = paddle.equal(node_x, node_y)
        bool_y = paddle.equal(node_x, node_z)

        out = paddle.logical_and(bool_x, bool_y)
        out = paddle.cast(out, x.dtype)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
            
        outs = exe.run(
            feed={'x': x, 'y': y, 'z': z},
            fetch_list=[out])
            
        saveModel(name, exe, feed_vars=[node_x, node_y, node_z], fetchlist=[out],
            inputs=[x, y, z], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data_x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
    data_y = np.array([[[[2, 0, 3]], [[3, 1, 4]]]]).astype(np.float32)
    data_z = np.array([[[[1, 0, 5]], [[2, 1, 0]]]]).astype(np.float32)

    equal_logical_or("logical_or", data_x, data_y, data_z)



if __name__ == "__main__":
    main()
