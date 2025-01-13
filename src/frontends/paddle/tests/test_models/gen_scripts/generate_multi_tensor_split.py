# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import paddle
import numpy as np
import sys
import os

from save_model import saveModel

if paddle.__version__ >= '2.6.0':
    import paddle.base as fluid
else:
    from paddle import fluid

def create_multi_output_model():
    paddle.enable_static()

    # paddle model creation and inference
    num_splits = 20
    inp_blob_1 = np.random.randn(2, num_splits, 4, 4).astype(np.float32)

    if paddle.__version__ >= '2.0.0':
        x = paddle.static.data(name='x', shape=[2, num_splits, 4, 4], dtype='float32')
        test_layer = paddle.split(x, num_or_sections=num_splits, axis=1)
    else:
        x = fluid.data(name='x', shape=[2, num_splits, 4, 4], dtype='float32')
        test_layer = paddle.split(x, num_or_sections=num_splits, axis=1)
    
    var = []
    for i in range(num_splits//2):
        if paddle.__version__ >= '2.0.0':
            add = paddle.add(test_layer[2*i], test_layer[2*i+1])
        else:
            add = fluid.layers.elementwise_add(test_layer[2*i], test_layer[2*i+1])
        var.append(add)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())
    inp_dict = {'x': inp_blob_1}
    res_paddle = exe.run(fluid.default_main_program(), fetch_list=var, feed=inp_dict)

    saveModel("multi_tensor_split", exe, feed_vars=[x], fetchlist=var, inputs=[inp_blob_1], outputs=res_paddle, target_dir=sys.argv[1])

create_multi_output_model()

