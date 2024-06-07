# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# slice paddle model generator
#
import sys
import os

import numpy as np
import paddle

from save_model import exportModel
from save_model import saveModel

data_type = 'float32'

def slice(name : str, x, axes : list, start : list, end : list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype = data_type)
        out = paddle.slice(node_x, axes = axes, starts = start, ends = end)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def slice_dyn(test_shape=[2,8,10,10]):
    paddle.disable_static()

    data = paddle.rand(shape=test_shape, dtype='float32')

    '''
    slice w/ decrease_axis
    '''
    @paddle.jit.to_static
    def test_slice_decrease_axis(x):
        return x[0, 1:3, :, 5]
    exportModel('slice_decrease_axis', test_slice_decrease_axis, [data], target_dir=sys.argv[1]) # output shape (2, 10)

    '''
    slice w/o decrease_axis
    '''
    @paddle.jit.to_static
    def test_slice(x):
        return paddle.slice(x, axes=[0,1,3], starts=[0,1,5], ends=[1,3,6])
    # exportModel('slice_dyn', test_slice, [data], target_dir=sys.argv[1]) # output shape (1, 2, 10, 1)  # disable it by default as this kind of test model already there. It's for comparsion only.

    '''
    slice w/ decrease_axis of all dims
    '''
    @paddle.jit.to_static
    def test_slice_decrease_axis_all(x):
        return x[0, 0, 0, 0]
    exportModel('slice_decrease_axis_all', test_slice_decrease_axis_all, [data], target_dir=sys.argv[1]) # output shape (1,)

    '''
    slice w/o decrease_axis of all dims
    '''
    @paddle.jit.to_static
    def test_slice_alldim(x):
        return paddle.slice(x, axes=[0,1,2,3], starts=[0,0,0,0], ends=[1,1,1,1])
    # exportModel('slice_alldim', test_slice_alldim, [data], target_dir=sys.argv[1]) # output shape (1, 1, 1, 1) # disable it by default as this kind of test model already there. It's for comparsion only.

'''
a test case simulating the last reshape2 of ocrnet which accepts slice (with decrease_axes in all dims) as its parents.
'''
def slice_reshape(B=1, C=256, H=16, W=32):
    paddle.disable_static()

    data = paddle.rand(shape=[B, C, H*W], dtype='float32')

    @paddle.jit.to_static
    def test_model(x):
        x2 = paddle.assign([-1, -1, 16, 32]).astype('int32')
        node_reshape = paddle.reshape(x, [0, 256, x2[2], x2[3]])
        return node_reshape
    exportModel('slice_reshape', test_model, [data], target_dir=sys.argv[1])

def main():
    x = np.linspace(1, 60, num = 60, dtype=np.int32).reshape(4, 3, 5).astype(data_type)
    slice("slice", x, axes=[1, 2], start=(0, 1), end=(-1, 3))

    x = np.linspace(1, 60, num = 60, dtype=np.int32).reshape(2, 30).astype(data_type)
    slice("slice_1d", x, axes=[0], start=[0], end=[1])

if __name__ == "__main__":
    main()
    slice_dyn()
    slice_reshape()
