# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# fill_const paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def eye(name : str, rows, cols = None, dtype = None):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        if paddle.__version__ >= '2.0.0':
            x1 = paddle.eye(num_rows=rows, num_columns=cols, dtype=dtype, name='fill')
            x2 = paddle.eye(num_rows=rows, num_columns=cols, dtype=dtype, name='fill')
        else:
            x1 = paddle.fluid.layers.eye(num_rows=rows, num_columns=cols, dtype=dtype, name='fill_constant')
            x2 = paddle.fluid.layers.eye(num_rows=rows, num_columns=cols, dtype=dtype, name='fill_constant')
        out = paddle.add(x1, x2)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            fetch_list=[out])             

        saveModel(name, exe, feed_vars=[], fetchlist=[out], inputs=[], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    eye("eye", 3)
    eye("eye_int32", 2, 3, "int32")
    eye("eye_int64", 2, 3, "int64")

if __name__ == "__main__":
    main()
