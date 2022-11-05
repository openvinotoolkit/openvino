# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import paddle
from paddle import fluid
import numpy as np
import os
import sys

paddle.enable_static()

inp_blob = np.random.randn(1, 3, 4, 4).astype(np.float32)

x = fluid.data(name='x', shape=[1, 3, 4, 4], dtype='float32')
test_layer = fluid.layers.conv2d(input=x, num_filters=5, filter_size=(1, 1), stride=(1, 1), padding=(1, 1),
                                 dilation=(1, 1), groups=1, bias_attr=False)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
inp_dict = {'x': inp_blob}
var = [test_layer]
res_paddle = exe.run(fluid.default_main_program(), fetch_list=var, feed=inp_dict)

fluid.io.save_inference_model(os.path.join(sys.argv[1], "conv2d_s"), list(inp_dict.keys()), var, exe,
                              model_filename="conv2d.pdmodel", params_filename="conv2d.pdiparams")
