# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# mean paddle model generator
#

import numpy as np
import sys
from save_model import saveModel


def mean(name : str, x, axis=None, keepdim=False):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = paddle.sum(data_x, axis=axis, keepdim=keepdim)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
        feed={'x': x},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.random.random((2,3,4)).astype(np.float32)

    mean("mean_1", data, axis=0, keepdim=False)
    mean("mean_2", data, axis=-1, keepdim=False)
    mean("mean_3", data, axis=1, keepdim=True)
    mean("mean_4", data, axis=[0,1], keepdim=True)


if __name__ == "__main__":
    main()
