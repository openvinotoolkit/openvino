# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# relu paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def prelu(name: str, x, alpha, data_format='NCHW'):
    import paddle
    paddle.enable_static()

    node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
    node_alpha = paddle.static.data(name='alpha', shape=alpha.shape, dtype='float32')

    out = paddle.nn.functional.prelu(node_x, node_alpha, data_format)

    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(paddle.static.default_startup_program())

    outs = exe.run(
        feed={'x': x,'alpha':alpha},
        fetch_list=[out])

    saveModel(name, exe, feedkeys=['x','alpha'], fetchlist=[out],
              inputs=[x,alpha], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.array([-2, 0, 1]).astype('float32')
    weight= np.array([0.25]).astype('float32')
    prelu("prelu", data, weight)


if __name__ == "__main__":
    main()
