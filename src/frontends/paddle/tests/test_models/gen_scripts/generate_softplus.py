# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# softplus paddle model generator
#
import numpy as np
import sys
from save_model import saveModel


def softplus(name: str, x, beta, threshold):
    import paddle
    paddle.enable_static()

    node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
    out = paddle.nn.functional.softplus(x=node_x, beta=beta, threshold=threshold)

    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(paddle.static.default_startup_program())

    outs = exe.run(
        feed={'x': x},
        fetch_list=[out])

    saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.array(
        [[[2.0, 3.0, 4.0, 5.0],
          [3.0, 4.0, 5.0, 6.0],
          [7.0, 8.0, 8.0, 9.0]],
         [[1.0, 2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0, 8.0],
          [6.0, 7.0, 8.0, 9.0]]]
    ).astype(np.float32)

    softplus("softplus_default_params", data, beta=1, threshold=20)

if __name__ == "__main__":
    main()
