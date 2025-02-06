# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# fill_const paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def take_along_axis(name : str, x,index,axis):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_index = paddle.static.data(name='index', shape=index.shape, dtype=index.dtype)
        out=paddle.take_along_axis(node_x, node_index, axis)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x,'index':index},
            fetch_list=[out])             
        saveModel(name, exe, feed_vars=[node_x,node_index], fetchlist=[out], inputs=[x,index], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    x = np.array([[1, 2, 3], [4, 5, 6], [7,8,9]]).astype(np.float32)
    index = np.array([[0]]).astype(np.int64)
    axis=0
    take_along_axis("take_along_axis_test_1", x,index,axis)



if __name__ == "__main__":
    main()