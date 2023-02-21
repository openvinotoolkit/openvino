# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from save_model import saveModel
import sys


def paddle_mean(name: str, x, axis, keepdim):
    import paddle

    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype="float32")
        out = paddle.mean(node_x, axis=axis, keepdim=keepdim, name="mean")
        out = paddle.cast(out, "float32")
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


if __name__ == "__main__":
    x = np.random.random([3, 5, 6, 9]).astype(np.float32)
    paddle_mean("mean_negative_axis", x, axis=-1, keepdim=False)
    paddle_mean("mean_negative_axis_keepdim", x, axis=-1, keepdim=True)
    paddle_mean("mean_none_axis", x, axis=None, keepdim=False)
    paddle_mean("mean_none_axis_keepdim", x, axis=None, keepdim=True)
    paddle_mean("mean_positive_axis", x, axis=1, keepdim=False)
    paddle_mean("mean_list_axis", x, axis=[1, 2], keepdim=False)
    paddle_mean("mean_list_axis_keepdim", x, axis=[1, 2], keepdim=True)
