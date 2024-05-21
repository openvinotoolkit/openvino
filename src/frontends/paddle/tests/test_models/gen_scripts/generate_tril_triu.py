# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# tril_triu ops paddle model generator
#

import sys

import numpy as np
import paddle
from save_model import saveModel


def triu(name: str, x, diagonal=0, dtype="float32"):
    paddle.enable_static()
    x = x.astype(dtype)

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=dtype)

        triu_outs = paddle.triu(node_x, diagonal)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x}, fetch_list=[triu_outs])

        saveModel(
            name,
            exe,
            feed_vars=[node_x],
            fetchlist=[triu_outs],
            inputs=[x],
            outputs=outs,
            target_dir=sys.argv[1],
        )


def tril(name: str, x, diagonal=0, dtype="float32"):
    paddle.enable_static()
    x = x.astype(dtype)

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=dtype)

        tril_outs = paddle.tril(node_x, diagonal)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x}, fetch_list=[tril_outs])

        saveModel(
            name,
            exe,
            feed_vars=[node_x],
            fetchlist=[tril_outs],
            inputs=[x],
            outputs=outs,
            target_dir=sys.argv[1],
        )


def main():
    data = np.random.randn(4, 5) * 10
    triu("triu", data)
    triu("triu_1", data, 1)
    triu("triu_2", data, -1)
    triu("triu_3", data, 10)
    triu("triu_4", data, -3)
    triu("triu_int32", data, dtype="int32")
    triu("triu_int64", data, dtype="int64")

    tril("tril", data)
    tril("tril_1", data, 1)
    tril("tril_2", data, -1)
    tril("tril_3", data, 10)
    tril("tril_4", data, -3)
    tril("tril_int32", data, dtype="int32")
    tril("tril_int64", data, dtype="int64")


if __name__ == "__main__":
    main()
