# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# where paddle model generator
#
import numpy as np
from save_model import saveModel
import sys
import paddle

paddle.enable_static()


def where_index(name: str, x, force_boolean=False):
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        if force_boolean:
            if paddle.__version__ >= '2.0.0':
                node_x_bl = paddle.cast(node_x, "bool")
            else:
                node_x_bl = paddle.fluid.layers.cast(node_x, "bool")
            out = paddle.nonzero(node_x_bl)
        else:
            out = paddle.nonzero(node_x)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[
                  x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    # case of int32
    datatype = "int32"
    condition = np.random.randint(0, 5, size=[5, 8, 2], dtype=datatype)
    paddle_out = where_index("where_index_1", condition)

    # case of float32
    datatype = "float32"
    condition = (np.random.randint(
        0, 5, size=[8, 3, 2]) * 1.1).astype(datatype)
    paddle_out = where_index("where_index_2", condition)

    # case of dimension 4
    condition = (np.random.randint(
        0, 5, size=[8, 3, 2, 6]) * 1.1).astype(datatype)
    paddle_out = where_index("where_index_3", condition)

    # case of dimension 5
    condition = (np.random.randint(
        0, 5, size=[4, 6, 8, 2, 5]) * 1.1).astype(datatype)
    paddle_out = where_index("where_index_4", condition)

    # case of rank 1
    condition = np.ones(10).astype(datatype)
    paddle_out = where_index("where_index_5", condition, force_boolean=True)

    # case of rank 1 and boolean zeros
    condition = np.array([1, 0, 1]).astype(datatype)
    paddle_out = where_index("where_index_6", condition, force_boolean=True)


if __name__ == "__main__":
    main()
