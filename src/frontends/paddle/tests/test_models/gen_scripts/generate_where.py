# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# where paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def where(name, test_x, test_y, test_cond):
    import paddle
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        X_Node = paddle.static.data(
            name='x', shape=test_x.shape, dtype=test_x.dtype)
        Y_Node = paddle.static.data(
            name='y', shape=test_y.shape, dtype=test_y.dtype)
        Cond_Node = paddle.static.data(
            name='cond', shape=test_cond.shape, dtype=test_cond.dtype)

        if paddle.__version__ >= '2.0.0':
            Cond_Node_bl = paddle.cast(Cond_Node, "bool")
        else:
            Cond_Node_bl = paddle.fluid.layers.cast(Cond_Node, "bool")

        out = paddle.where(Cond_Node_bl, X_Node, Y_Node)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': test_x, 'y': test_y, 'cond': test_cond},
            fetch_list=[out]
        )

        saveModel(name, exe, feed_vars=[X_Node, Y_Node, Cond_Node], fetchlist=[out], inputs=[
                  test_x, test_y, test_cond], outputs=[outs[0]], target_dir=sys.argv[1])


def main():

    test_cases = [
        {
            "name": "where_1",
            "x": np.random.uniform(-3, 5, (100)).astype("float32"),
            "y": np.random.uniform(-3, 5, (100)).astype("float32"),
            "cond": np.zeros((100)).astype("int32")
        },
        {
            "name": "where_2",
            "x": np.random.uniform(-5, 5, (60, 2)).astype("int32"),
            "y": np.random.uniform(-5, 5, (60, 2)).astype("int32"),
            "cond": np.ones((60, 2)).astype("int32")
        },
        {
            "name": "where_3",
            "x": np.random.uniform(-3, 5, (20, 2, 4)).astype("float32"),
            "y": np.random.uniform(-3, 5, (20, 2, 4)).astype("float32"),
            "cond": np.array(np.random.randint(2, size=(20, 2, 4)), dtype="int32")
        }
    ]
    for test in test_cases:
        where(test['name'], test['x'], test['y'], test['cond'])


if __name__ == "__main__":
    main()
