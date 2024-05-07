# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# tile paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def paddle_tile(name: str, x, repeat_times, to_tensor=False, tensor_list=False):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(
            name="x",
            shape=x.shape,
            dtype=x.dtype
        )
        node_x = paddle.cast(node_x, dtype=x.dtype)
        repeat_times_list = []
        if tensor_list:
            for i in repeat_times:
                if paddle.__version__ >= '2.0.0':
                    temp_out = paddle.full([1], i, "int32").cpu()
                else:
                    temp_out = paddle.fluid.layers.tensor.fill_constant([1], "int32", i, force_cpu=True)

                repeat_times_list.append(temp_out)
        else:
            repeat_times_list = repeat_times

        if to_tensor:
            repeat_times_list = paddle.static.data(
                name="repeat_times",
                shape=repeat_times.shape,
                dtype=repeat_times.dtype,
            )
        out = paddle.tile(node_x, repeat_times_list)

        if out.dtype == paddle.bool:
            out = paddle.cast(out, dtype=np.int32)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        if to_tensor:
            feed = {"x": x, "repeat_times": repeat_times}
            feed_vars = [node_x, repeat_times_list]
        else:
            feed = {"x": x}
            feed_vars = [node_x]

        outs = exe.run(feed=feed, fetch_list=[out])

        saveModel(
            name,
            exe,
            feed_vars=feed_vars,
            fetchlist=[out],
            inputs=[*feed.values()],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )
    return outs[0]


def main():
    test_cases = [
        "float32",
        "int32",
        "int64",
        "bool",
    ]
    for case in test_cases:
        x = np.array([1, 2, 3]).astype(case)
        paddle_tile("tile_list_" + case, x, [2, 1])
        paddle_tile("tile_tuple_" + case, x, (2, 2))

    x = np.array([1, 2, 1]).astype("float32")
    repeat_times = np.array([2, 1]).astype("int32")
    paddle_tile("tile_tensor_list", x, repeat_times, tensor_list=True)
    paddle_tile("tile_repeat_times_tensor", x, repeat_times, to_tensor=True)
    x = np.random.rand(2, 3, 4).astype("float32")
    paddle_tile("tile_repeat_gt_x", x, [5, 1, 2, 3])
    paddle_tile("tile_repeat_lt_x", x, [2, 3])


if __name__ == "__main__":
    main()
