# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# fill_const paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def reduce_any(name : str, x, axis=None, keepdim=False):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        if paddle.__version__ >= '2.0.0':
            cast_node=paddle.cast(node_x, dtype="bool")
            any_out = paddle.any(cast_node, axis=axis, keepdim=keepdim)
            out = paddle.cast(any_out, x.dtype)
        else:
            cast_node=paddle.fluid.layers.cast(node_x, "bool")
            any_out = paddle.fluid.layers.reduce_any(cast_node, axis=axis, keepdim=keepdim)
            out = paddle.cast(any_out, x.dtype)
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
    data = np.array([[1,0], [1, 1]]).astype(np.float32)
    reduce_any("reduce_any_test_0", data)
    reduce_any("reduce_any_test_1", data, axis=0, keepdim=False)
    reduce_any("reduce_any_test_2", data, axis=-1, keepdim=False)
    reduce_any("reduce_any_test_3", data, axis=1, keepdim=True)
    reduce_any("reduce_any_test_4", data, axis=[0,1], keepdim=True)

if __name__ == "__main__":
    main()