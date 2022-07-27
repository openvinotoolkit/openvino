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


def paddle_tile(name: str, x, repeat_times, to_tensor=False):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        if to_tensor:
            node_repeat_times = paddle.static.data(
                name="repeat_times", shape=repeat_times.shape, dtype="int32"
            )
        out = paddle.tile(node_x, repeat_times if not to_tensor else node_repeat_times)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        if to_tensor:
            feed = {"x": x, "repeat_times": repeat_times}
        else:
            feed = {"x": x}

        outs = exe.run(feed=feed, fetch_list=[out])

        saveModel(
            name,
            exe,
            feedkeys=[*feed.keys()],
            fetchlist=[out],
            inputs=[*feed.values()],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    x = np.array([1, 2, 3]).astype("float32")
    paddle_tile("tile_list", x, [2, 1])
    paddle_tile("tile_tuple", x, (2, 2))
    repeat_times = np.array([1, 2]).astype("int32")
    paddle_tile("tile_repeat_times_tensor", x, repeat_times, to_tensor=True)


if __name__ == "__main__":
    main()
