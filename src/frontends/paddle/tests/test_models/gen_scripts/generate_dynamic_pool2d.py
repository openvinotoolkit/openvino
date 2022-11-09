# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pool2d paddle dynamic model generator
#

import paddle
from paddle import fluid
import numpy as np
import sys
import os
from save_model import saveModel

paddle.enable_static()
inp_blob1 = np.random.randn(1, 1, 224, 224).astype(np.float32)

x1 = fluid.data(name='inputX1', shape=[1, 1, -1, -1], dtype='float32')

adative_pool2d = paddle.fluid.layers.adaptive_pool2d(
    input=x1,
    pool_size=[3,3],
    pool_type='avg',
    require_index=False)

cpu = paddle.static.cpu_places(1)
exe = paddle.static.Executor(cpu[0])
# startup program will call initializer to initialize the parameters.
exe.run(paddle.static.default_startup_program())

outs = exe.run(
    feed={'inputX1': inp_blob1},
    fetch_list=[adative_pool2d])

saveModel("pool2d_dyn_hw", exe, feedkeys=['inputX1'], fetchlist=adative_pool2d, inputs=[inp_blob1], outputs=outs, target_dir=sys.argv[1])
