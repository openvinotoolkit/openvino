# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pad3d paddle model generator
#
import numpy as np
from save_model import saveModel
import sys

def pad3d(name : str, x, in_dtype, pad, data_format, mode, value = 0):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name = 'x', shape = x.shape, dtype = in_dtype)

        if mode == 'constant':
            pad_constant = paddle.nn.Pad3D(padding=pad, mode=mode, value=value, data_format=data_format)
            out = pad_constant(node_x)
        else:
            pad_other_mode = paddle.nn.Pad3D(padding=pad, mode=mode, data_format=data_format)
            out = pad_other_mode(node_x)

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
    in_dtype = 'float32'

    input_shape = (1, 2, 3, 4, 5)
    pad = [1, 2, 1, 1, 3, 4]
    mode = 'constant'
    data_format = 'NCDHW'
    value = 100
    input_data = np.random.rand(*input_shape).astype(np.float32)
    pad3d("pad3d_test1", input_data, in_dtype, pad, data_format, mode, value)

    input_shape = (2, 3, 4, 5, 6)
    pad = [1, 2, 1, 1, 1, 2]
    mode = "reflect"
    data_format = 'NDHWC'
    input_data = np.random.rand(*input_shape).astype(np.float32)
    pad3d("pad3d_test2", input_data, in_dtype, pad, data_format, mode)

    input_shape = (2, 3, 4, 5, 6)
    pad = [1, 2, 1, 1, 1, 2]
    mode = "replicate"
    data_format = 'NDHWC'
    input_data = np.random.rand(*input_shape).astype(np.float32)
    pad3d("pad3d_test3", input_data, in_dtype, pad, data_format, mode)

    input_shape = (2, 3, 4, 5, 6)
    pad = [1, 2, 1, 1, 1, 2]
    mode = "circular"
    data_format = 'NDHWC'
    input_data = np.random.rand(*input_shape).astype(np.float32)
    pad3d("throw_in_conversion", input_data, in_dtype, pad, data_format, mode)
    # padding of type int feature only supported by PaddlePaddle 'develop' version(>=2.1.0)
#    input_shape = (1, 2, 3, 4, 5)
#    pad_int = 1
#    mode = 'constant'
#    data_format= 'NCDHW'
#    value = 100
#    input_data = np.random.rand(*input_shape).astype(np.float32)
#    pad3d("pad3d_test4", input_data, in_dtype, pad_int, data_format, mode, value)

if __name__ == "__main__":
    main()
