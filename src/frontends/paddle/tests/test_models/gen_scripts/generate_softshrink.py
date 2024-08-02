# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# softshrink paddle model generator
#
import numpy as np
import sys
from save_model import saveModel


def softshrink(name: str, x, threshold):
    import paddle

    paddle.enable_static()

    node_x = paddle.static.data(name="x", shape=x.shape, dtype="float32")
    if threshold == None:
        out = paddle.nn.functional.softshrink(node_x)
    else:
        out = paddle.nn.functional.softshrink(node_x, threshold)

    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(paddle.static.default_startup_program())

    outs = exe.run(feed={"x": x}, fetch_list=[out])

    saveModel(
        name,
        exe,
        feed_vars=[node_x],
        fetchlist=[out],
        inputs=[x],
        outputs=[outs[0]],
        target_dir=sys.argv[1],
    )

    return outs[0]


def main():
    data = np.array(
        [
            [[2.0, 3.0, 4.0, 5.0], [0.0, 4.0, -5.0, 6.0], [7.0, -8.0, 8.0, 9.0]],
            [[-1.0, 2.0, 3.0, 4.0], [-5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]],
        ]
    ).astype(np.float32)

    softshrink("softshrink_default_params", data, threshold=None)
    softshrink("softshrink_threshold_0.6", data, threshold=0.6)


if __name__ == "__main__":
    main()
