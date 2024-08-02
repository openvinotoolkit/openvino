# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
 
import os
import sys

import numpy as np
import paddle

from save_model import exportModel

'''
assign w/ output
'''
@paddle.jit.to_static
def test_assign_output(array):
    result1 = paddle.zeros(shape=[3, 2], dtype='float32')
    paddle.assign(array, result1) # result1 = [[1, 1], [3 4], [1, 3]]
    return result1

array = np.array([[1, 1],
                [3, 4],
                [1, 3]]).astype(np.int64)
exportModel('assign_output', test_assign_output, [array], target_dir=sys.argv[1])

'''
assign w/o output
'''
@paddle.jit.to_static
def test_assign_none(data):
    result2 = paddle.assign(data)  # result2 = [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
    return result2

data = paddle.full(shape=[3, 2], fill_value=2.5, dtype='float32') # [[2.5, 2.5], [2.5, 2.5], [2.5, 2.5]]
exportModel('assign_none', test_assign_none, [data], target_dir=sys.argv[1])
