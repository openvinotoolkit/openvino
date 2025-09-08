# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# fill_const paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def scatter(name : str, x,index,updates,overwrite=True):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_index = paddle.static.data(name='index', shape=index.shape, dtype=index.dtype)
        node_updates=paddle.static.data(name='updates', shape=updates.shape, dtype=updates.dtype)
        out=paddle.scatter(node_x, node_index, node_updates, overwrite)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x,'index':index,'updates':updates},
            fetch_list=[out])             
        saveModel(name, exe, feed_vars=[node_x,node_index,node_updates], fetchlist=[out], inputs=[x,index,updates], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    x = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float32)
    index = np.array([2, 1, 0, 1]).astype(np.int64)
    updates = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(np.float32)

    scatter("scatter_test_1", x,index,updates,overwrite=True)
    scatter("scatter_test_2", x,index,updates,overwrite=False)



if __name__ == "__main__":
    main()