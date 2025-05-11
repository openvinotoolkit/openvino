# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pool2d paddle model generator
#
import numpy as np
import sys
from save_model import saveModel


def paddle_scale(name : str, x, scale, bias, attrs : dict, data_type):
    import paddle
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        out = paddle.scale(x=node_x, scale=scale, bias=bias,
                         bias_after_scale=attrs['bias_after_scale'])
        #FuzzyTest only support FP32 now, so cast result to fp32
        out = paddle.cast(out, "float32")
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def paddle_scale_tensor(name : str, x, scale, bias, attrs : dict, data_type):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        if paddle.__version__ >= '2.5.1':
            node_scale = paddle.static.data(name='scale', shape=[], dtype='float32')
        else:
            node_scale = paddle.static.data(name='scale', shape=[1], dtype='float32')
        out = paddle.scale(x=node_x, scale=node_scale, bias=bias,
                         bias_after_scale=attrs['bias_after_scale'])
        #FuzzyTest only support FP32 now, so cast result to fp32
        out = paddle.cast(out, "float32")
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'scale': scale},
            fetch_list=[out])

        if paddle.__version__ >= '2.5.1':
            saveModel(name, exe, feed_vars=[node_x, node_scale], fetchlist=[out], inputs=[x, np.array(scale).astype('float32')], outputs=[outs[0]], target_dir=sys.argv[1])
        else:
            saveModel(name, exe, feed_vars=[node_x, node_scale], fetchlist=[out], inputs=[x, np.array([scale]).astype('float32')], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    scale = 2.0
    bias = 1.0
    data = np.random.random([2, 3]).astype("float32")

    test_cases = [
        "float32",
        "int32",
        "int64"
    ]

    paddle_attrs = {
        'bias_after_scale': True,
    }
    paddle_scale_tensor("scale_tensor_bias_after", data, scale, bias, paddle_attrs, 'float32')

    paddle_attrs = {
        'bias_after_scale': False,
    }
    paddle_scale_tensor("scale_tensor_bias_before", data, scale, bias, paddle_attrs, 'float32')

    for test in test_cases:
        data = np.random.random([2, 3]).astype(test)
        paddle_attrs = {
            'bias_after_scale': True,
        }
        paddle_scale("scale_bias_after_" + test, data, scale, bias, paddle_attrs, test)

        paddle_attrs = {
            'bias_after_scale': False,
        }
        paddle_scale("scale_bias_before_" + test, data, scale, bias, paddle_attrs, test)



if __name__ == "__main__":
    main()     
