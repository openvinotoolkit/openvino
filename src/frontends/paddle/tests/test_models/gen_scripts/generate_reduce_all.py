# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# reduce_all paddle model generator
#

import numpy as np
import sys
from save_model import saveModel


def reduce_all(name : str, x, axis=None, keepdim=False):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        if paddle.__version__ >= '2.0.0':
            reduced = paddle.all(data_x, axis=axis, keepdim=keepdim)
        else:
            reduced = paddle.fluid.layers.reduce_all(data_x, dim=axis, keep_dim=keepdim)
        out = paddle.cast(reduced, 'int32')

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
    sample_arr = [True, False]
    data = np.random.choice(sample_arr, size=(3,4,5))

    reduce_all("reduce_all_test_0", data)
    reduce_all("reduce_all_test_1", data, axis=0, keepdim=False)
    reduce_all("reduce_all_test_2", data, axis=-1, keepdim=False)
    reduce_all("reduce_all_test_3", data, axis=1, keepdim=True)
    reduce_all("reduce_all_test_4", data, axis=[1,2], keepdim=False)
    reduce_all("reduce_all_test_5", data, axis=[0,1], keepdim=True)


if __name__ == "__main__":
    main()
