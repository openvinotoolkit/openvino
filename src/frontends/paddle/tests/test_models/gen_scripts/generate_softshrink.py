# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# softshrink paddle model generator
#

import numpy as np
import sys
from save_model import saveModel


def softshrink(name : str, x, thres=0.5):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        func = paddle.nn.Softshrink(threshold=thres)
        out = func.forward(data)

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
    data = np.random.random((2,3)).astype(np.float32)

    softshrink("softshrink_1", data)
    softshrink("softshrink_2", data, thres=0)
    softshrink("softshrink_3", data, thres=1)


if __name__ == "__main__":
    main()
