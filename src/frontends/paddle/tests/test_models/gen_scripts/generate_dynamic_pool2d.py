# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pool2d paddle dynamic model generator
#

import paddle
import numpy as np
import sys
import os
from save_model import saveModel

if paddle.__version__ >= '2.6.0':
    import paddle.base as fluid
else:
    from paddle import fluid

paddle.enable_static()
inp_blob1 = np.random.randn(1, 1, 224, 224).astype(np.float32)

if paddle.__version__ >= '2.0.0':
    x1 = paddle.static.data(name='inputX1', shape=[
                            1, 1, -1, -1], dtype='float32')

    adaptive_pool2d = paddle.nn.functional.adaptive_avg_pool2d(
        x=x1,
        output_size=[3, 3])
else:
    x1 = fluid.data(name='inputX1', shape=[1, 1, -1, -1], dtype='float32')

    adaptive_pool2d = paddle.fluid.layers.adaptive_pool2d(
        input=x1,
        pool_size=[3, 3],
        pool_type='avg',
        require_index=False)

cpu = paddle.static.cpu_places(1)
exe = paddle.static.Executor(cpu[0])
# startup program will call initializer to initialize the parameters.
exe.run(paddle.static.default_startup_program())

outs = exe.run(
    feed={'inputX1': inp_blob1},
    fetch_list=[adaptive_pool2d])

saveModel("pool2d_dyn_hw", exe, feed_vars=[x1], fetchlist=adaptive_pool2d, inputs=[
          inp_blob1], outputs=outs, target_dir=sys.argv[1])
