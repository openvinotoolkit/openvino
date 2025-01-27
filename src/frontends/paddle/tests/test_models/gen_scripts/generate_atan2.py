# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# fill_const paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def atan2(name , x , y):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=y.dtype)
        atan2_node = paddle.atan2(node_x,node_y, name='atan2_node')
        out = paddle.assign(atan2_node)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])             
        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out], inputs=[x,y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    if paddle.__version__ >= '2.0.0':
        input_x = np.array([-1, 1, 1, -1]).astype(np.float32)
        input_y = np.array([-1, -1, 1, 1]).astype(np.float32)
        atan2("atan2",input_x,input_y)

if __name__ == "__main__":
    main()