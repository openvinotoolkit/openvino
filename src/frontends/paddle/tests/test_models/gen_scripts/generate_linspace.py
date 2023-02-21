# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# linspace paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import random
import sys

data_type = "float32"


def linespace(name: str, start, stop, num, type=None):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        if isinstance(start, np.ndarray):
            in_start = paddle.static.data(name="start", shape=[1], dtype=type)
            in_stop = paddle.static.data(name="stop", shape=[1], dtype=type)
            in_num = paddle.static.data(name="num", shape=[1], dtype='int32')
        else:
            in_start = paddle.full(name="start", shape=[1], fill_value=start, dtype=type)
            in_stop = paddle.full(name="stop", shape=[1], fill_value=stop, dtype=type)
            in_num = paddle.full(name="num", shape=[1], fill_value=num, dtype='int32')

        out = paddle.linspace(in_start, in_stop, in_num)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"Start": in_start, "Stop": in_stop, "Num": in_num}, fetch_list=[out])

        saveModel(name, exe, feedkeys=["Start", "Stop", "Num"], fetchlist=[out], inputs=[in_start, in_stop, in_num], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    start = random.randint(0, 10)
    stop = random.randint(0, 10)
    num = random.randint(1, 10)
    linespace("linespace_1", start, stop, num, "int32")

    start = random.uniform(0, 10)
    stop = random.uniform(0, 10)
    num = random.randint(1, 10)
    linespace("linespace_2", start, stop, num, "float32")

    # start = np.random.randn(1).astype(np.float32)
    # stop = np.random.randn(1).astype(np.float32)
    # num = np.random.randint(1, 5)
    # linespace("linespace_3", start, stop, num, "float32")


if __name__ == "__main__":
    main()
