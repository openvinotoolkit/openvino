# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import paddle
import numpy as np
import sys

from save_model import saveModel 

if paddle.__version__ >= '2.6.0':
    import paddle.base as fluid
else:
    from paddle import fluid

paddle.enable_static()

inp_blob = np.random.randn(1, 3, 4, 4).astype(np.float32)

if paddle.__version__ >= '2.0.0':
    x = paddle.static.data(name='xxx', shape=[1, 3, 4, 4], dtype='float32')
    test_layer = paddle.static.nn.conv2d(input=x, num_filters=5, filter_size=(1, 1), stride=(1, 1), padding=(1, 1),
                                         dilation=(1, 1), groups=1, bias_attr=False)
else:
    x = fluid.data(name='xxx', shape=[1, 3, 4, 4], dtype='float32')
    test_layer = fluid.layers.conv2d(input=x, num_filters=5, filter_size=(1, 1), stride=(1, 1), padding=(1, 1),
                                 dilation=(1, 1), groups=1, bias_attr=False)

if paddle.__version__ >= '2.0.0':
    relu = paddle.nn.functional.relu(test_layer)
else:
    relu = fluid.layers.relu(test_layer)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
inp_dict = {'xxx': inp_blob}
var = [relu]
res_paddle = exe.run(fluid.default_main_program(),
                     fetch_list=var, feed=inp_dict)

saveModel("conv2d_relu", exe, [x], var, [inp_blob], [res_paddle[0]], target_dir=sys.argv[1])
