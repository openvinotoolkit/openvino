# Copyright (C) 2018-2025 Intel Corporation
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


def linspace(name: str, start, stop, num, type='float32'):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data_start = paddle.static.data(name="Start", shape=[1], dtype=start.dtype)
        data_stop = paddle.static.data(name="Stop", shape=[1], dtype=stop.dtype)
        data_num = paddle.static.data(name="Num", shape=[1], dtype='int32')

        out = paddle.linspace(data_start, data_stop, data_num, dtype=type)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"Start": start, "Stop": stop, "Num": num}, fetch_list=[out])

        saveModel(name, exe, feed_vars=[data_start, data_stop, data_num], fetchlist=[out], inputs=[start, stop, num],
                  outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    # random test, float32
    start = np.random.randn(1).astype(np.float32)
    stop = np.random.randn(1).astype(np.float32)
    num = np.random.randint(1, 5, size=1).astype(np.int32)
    linspace("linspace_1", start, stop, num, "float32")
    # int32 to float32
    start = np.array([0]).astype(np.int32)
    stop = np.array([1]).astype(np.int32)
    num = np.array([4]).astype(np.int32)
    linspace("linspace_2", start, stop, num, "float32")
    # int64, start less than stop, minimal num = 1
    start = np.array([-5]).astype(np.int64)
    stop = np.array([-4]).astype(np.int64)
    num = np.array([1]).astype(np.int32)
    linspace("linspace_3", start, stop, num, "int64")


if __name__ == "__main__":
    main()
