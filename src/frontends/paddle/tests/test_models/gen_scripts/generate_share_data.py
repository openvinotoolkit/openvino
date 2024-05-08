# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# share_data paddle model generator
#

import numpy as np
import sys
from save_model import saveModel


def share_data(name : str, x, axis=None, keepdim=False):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = data_x.detach()

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
    data = np.random.rand(3,4,5).astype("float32")

    share_data("share_data_test_0", data)


if __name__ == "__main__":
    main()
