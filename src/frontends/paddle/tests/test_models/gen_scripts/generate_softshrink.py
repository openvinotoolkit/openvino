# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# softshrink paddle model generator
#
import paddle
import numpy as np
import sys
from save_model import saveModel


def softshrink(name : str, x, thres=0.5):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = paddle.nn.functional.softshrink(data, threshold=thres)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        exe.run(paddle.static.default_startup_program())
        outs = exe.run(feed={'x': x}, fetch_list=[out])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.random.random((12,10)).astype(np.float32)
    softshrink("softshrink_1", data)
    data = np.random.random((12,10)).astype(np.float32)
    softshrink("softshrink_2", data, thres=0)


if __name__ == "__main__":
    main()
