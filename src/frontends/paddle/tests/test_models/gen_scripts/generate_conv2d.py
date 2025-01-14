# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import sys
from save_model import saveModel

def conv2d(name: str, x, dtype):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
    # inp_blob = np.random.randn(1, 3, 4, 4).astype(np.float32)

        node_x = paddle.static.data(name='x', shape=[1, 3, 4, 4], dtype='float32')
        conv2d_layer = paddle.nn.Conv2D(in_channels=3, out_channels=5, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1),
                                        dilation=(1, 1), groups=1, bias_attr=False)
        out = conv2d_layer(node_x)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        exe.run(paddle.static.default_startup_program())

        inp_dict = {'x': x}
        var = [out]
        outs = exe.run(feed=inp_dict, fetch_list=var)

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]

def main():
    dtype = "float32"
    data = np.random.randn(1, 3, 4, 4).astype(dtype)
    conv2d("conv2d", data, dtype)

if __name__ == "__main__":
    main()