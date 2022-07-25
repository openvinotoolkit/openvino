# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# tile paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = "float32"


def paddle_tile(name: str, x, repeat_times):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        out = paddle.tile(node_x, repeat_times)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feedkeys=["x"],
            fetchlist=[out],
            inputs=[x],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def paddle_tile_tensor(name: str, x, repeat_times):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        node_repeat_times = paddle.static.data(
            name="repeat_times", shape=repeat_times.shape, dtype="int32"
        )
        out = paddle.tile(node_x, node_repeat_times)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x, "repeat_times": repeat_times}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feedkeys=["x", "repeat_times"],
            fetchlist=[out],
            inputs=[x, repeat_times],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    x = np.array([1, 2, 3]).astype("float32")
    paddle_tile("tile_list", x, [2, 1])
    paddle_tile("tile_tuple", x, (2, 2))

    repeat_times = np.array([1, 2]).astype("int32")
    paddle_tile_tensor("tile_repeat_times_tensor", x, repeat_times)


if __name__ == "__main__":
    main()
