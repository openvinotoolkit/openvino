# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# layer_norm paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

if paddle.__version__ >= '2.6.0':
    from paddle.base import param_attr
else:
    from paddle.fluid import param_attr

data_type = 'float32'

def layer_norm(name:str, x, begin_norm_axis, scale=True, shift=True, param_attr=None, bias_attr=None):
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name='x', shape=x.shape, dtype = data_type)
        out = paddle.static.nn.layer_norm(input=data, scale=scale, shift=shift,\
            begin_norm_axis=begin_norm_axis, param_attr=param_attr, bias_attr=bias_attr)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[data], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    x = np.random.rand(8, 24, 32).astype(data_type)
    random_data = np.random.rand(24 * 32).astype(data_type)
    if paddle.__version__ >= '2.0.0':
        attr = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(random_data))
    else:
        attr = paddle.ParamAttr(
                initializer=paddle.fluid.initializer.NumpyArrayInitializer(random_data))
    layer_norm("layer_norm", x, begin_norm_axis=1, param_attr=attr, bias_attr=attr)
    layer_norm("layer_norm_noscale", x, scale=False, begin_norm_axis=2)
    layer_norm("layer_norm_noshift", x, shift=False, begin_norm_axis=1)
    layer_norm("layer_norm_noall", x, scale=False, shift=False, begin_norm_axis=1)

if __name__ == "__main__":
    main()
