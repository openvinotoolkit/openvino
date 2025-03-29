# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# p_norm paddle model generator
#

import sys

import numpy as np
import paddle

if paddle.__version__ >= '2.6.0':
    from paddle.base.layer_helper import LayerHelper
else:
    from paddle.fluid.layer_helper import LayerHelper

from save_model import saveModel

paddle.enable_static()


def p_norm_ref(x, p=None, axis=None, epsilon=1e-12, keepdim=None, name=None):
    attrs = {
        'axis': axis,
        'porder': p,
        'keepdim': keepdim,
        'epsilon': epsilon,
    }
    helper = LayerHelper('p_norm', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(type='p_norm',
                     inputs={'X': x},
                     outputs={'Out': out},
                     attrs=attrs)
    return out


def p_norm(name: str, x, axis, p, keepdim):
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)

        out = p_norm_ref(node_x, axis=axis, p=p, keepdim=keepdim)
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
    input_shape = (2, 3, 4, 5, 6)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    paddle_result = p_norm('p_norm1', input_data, axis=4, p=1.5, keepdim=True)

    input_shape = (3, 4, 5)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    paddle_result = p_norm('p_norm2', input_data, axis=0, p=0.0, keepdim=None)

    input_shape = (4, 5, 6)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    paddle_result = p_norm('p_norm3', input_data, axis=None, p=None, keepdim=True)

    input_shape = (6, 3, 4)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    paddle_result = p_norm('p_norm4', input_data, axis=1, p=float('inf'), keepdim=False)

    input_shape = (3, 5, 6)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    paddle_result = p_norm('p_norm5', input_data, axis=1, p=float('-inf'), keepdim=True)
    
    input_shape = (3, 6, 7)
    input_data = np.zeros(input_shape).astype(np.float32)
    paddle_result = p_norm('p_norm6', input_data, axis=0, p=0.0, keepdim=None)

    input_shape = (10)
    input_data = np.random.rand(input_shape).astype("float32")
    input_data[0:10:2] = 0
    paddle_result = p_norm('p_norm7', input_data, axis=0, p=0.0, keepdim=False)

    input_data = np.array([[0, 1, 2, -10]]).astype("float32")
    paddle_result = p_norm('p_norm8', input_data, axis=1, p=0.0, keepdim=False)

if __name__ == "__main__":
    main()
