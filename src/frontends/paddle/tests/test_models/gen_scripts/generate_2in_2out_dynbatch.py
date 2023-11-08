# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import paddle
from paddle import fluid
import numpy as np
import os
import sys


paddle.enable_static()

inp_blob1 = np.random.randn(1, 1, 3, 3).astype(np.float32)
inp_blob2 = np.random.randn(1, 2, 3, 3).astype(np.float32)

if paddle.__version__ >= '2.0.0':
    x1 = paddle.static.data(name='inputX1', shape=[-1, 1, 3, 3], dtype='float32')
    x2 = paddle.static.data(name='inputX2', shape=[-1, 2, 3, 3], dtype='float32')

    conv2d1 = paddle.static.nn.conv2d(input=x1, num_filters=1, filter_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                    dilation=(1, 1), groups=1, bias_attr=False, name="conv2dX1")

    conv2d2 = paddle.static.nn.conv2d(input=x2, num_filters=1, filter_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                dilation=(1, 1), groups=1, bias_attr=False, name="conv2dX2")

    add1 = paddle.add(conv2d1, conv2d2, name="add1")

    relu2a = paddle.nn.functional.relu(add1, name="relu2a")
    relu2b = paddle.nn.functional.relu(add1, name="relu2b")

    add2 = paddle.add(relu2a, relu2b, name="add2")

    relu3a = paddle.nn.functional.relu(add2, name="relu3a")
    relu3b = paddle.nn.functional.relu(add2, name="relu3b")
else:
    x1 = fluid.data(name='inputX1', shape=[-1, 1, 3, 3], dtype='float32')
    x2 = fluid.data(name='inputX2', shape=[-1, 2, 3, 3], dtype='float32')

    conv2d1 = fluid.layers.conv2d(input=x1, num_filters=1, filter_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                    dilation=(1, 1), groups=1, bias_attr=False, name="conv2dX1")

    conv2d2 = fluid.layers.conv2d(input=x2, num_filters=1, filter_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                dilation=(1, 1), groups=1, bias_attr=False, name="conv2dX2")

    add1 = fluid.layers.elementwise_add(conv2d1, conv2d2, name="add1")

    relu2a = fluid.layers.relu(add1, name="relu2a")
    relu2b = fluid.layers.relu(add1, name="relu2b")

    add2 = fluid.layers.elementwise_add(relu2a, relu2b, name="add2")

    relu3a = fluid.layers.relu(add2, name="relu3a")
    relu3b = fluid.layers.relu(add2, name="relu3b")

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
inp_dict = {'inputX1': inp_blob1, 'inputX2': inp_blob2}
var = [relu3a, relu3b]
res_paddle = exe.run(fluid.default_main_program(), fetch_list=var, feed=inp_dict)

fluid.io.save_inference_model(os.path.join(sys.argv[1], "2in_2out_dynbatch"), list(inp_dict.keys()), var, exe,
                              model_filename="2in_2out_dynbatch.pdmodel", params_filename="2in_2out_dynbatch.pdiparams")
