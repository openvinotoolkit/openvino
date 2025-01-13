# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# reduce_mean paddle model generator
#

import numpy as np
import sys
from save_model import saveModel


def reduce_mean(name : str, x, axis=None, keepdim=False):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = paddle.mean(data_x, axis=axis, keepdim=keepdim)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
        feed={'x': x},
        fetch_list=[out])
        saveModel(name, exe, feed_vars=[data_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.array([[[1.0,2.0], [3.0, 4.0]], [[5.0,6.0], [7.0, 8.0]]]).astype(np.float32)

    reduce_mean("reduce_mean_test_0", data)
    reduce_mean("reduce_mean_test_1", data, axis=0, keepdim=False)
    reduce_mean("reduce_mean_test_2", data, axis=-1, keepdim=False)
    reduce_mean("reduce_mean_test_3", data, axis=1, keepdim=True)
    reduce_mean("reduce_mean_test_4", data, axis=[1,2], keepdim=False)
    reduce_mean("reduce_mean_test_5", data, axis=[0,1], keepdim=True)


if __name__ == "__main__":
    main()
