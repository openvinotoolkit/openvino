# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# generate_flatten_contiguous_range paddle model generator
#
import numpy as np
from save_model import saveModel
import sys

def generate_flatten_contiguous_range(name : str, x, start_axis, stop_axis, in_dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name = 'x', shape = x.shape, dtype = in_dtype)
        out = paddle.flatten(node_x, start_axis, stop_axis)

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
    # TODO: more type
    in_dtype = 'float32'
    data = np.random.randn(3, 2, 5, 4).astype(in_dtype)
    start_axis = 1
    stop_axis = 2
    generate_flatten_contiguous_range("flatten_contiguous_range_test1", data, start_axis, stop_axis, in_dtype)

if __name__ == "__main__":
    main()
