# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# fill_const paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def scatter_nd_add(name : str, x,index,updates):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_index = paddle.static.data(name='index', shape=index.shape, dtype=index.dtype)
        node_updates=paddle.static.data(name='updates', shape=updates.shape, dtype=updates.dtype)
        out=paddle.scatter_nd_add(node_x, node_index, node_updates)
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
    x=np.random.rand(3, 5, 9, 10).astype(np.float32)
    index = np.array([[1, 1],[0, 1],[1, 3]]).astype(np.int64)
    updates = np.random.rand(3, 9, 10).astype(np.float32)
    scatter_nd_add("scatter_nd_add_test_1", x,index,updates)



if __name__ == "__main__":
    main()