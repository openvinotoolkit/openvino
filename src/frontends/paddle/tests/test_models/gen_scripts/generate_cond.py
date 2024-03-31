import os
import sys

import numpy as np
import paddle

from save_model import exportModel

'''
test: simple conditional_block pair + select_input without input to conditional_block.
'''
x = np.full(shape=[1], dtype='float32', fill_value=0.1)
y = np.full(shape=[1], dtype='float32', fill_value=0.23)
data = np.less(y,x)

@paddle.jit.to_static
def test_model(pred):
    # pred: A boolean tensor whose numel should be 1.
    def true_func():
        return paddle.full(shape=[3, 4], dtype='float32', # TODO: FAILED with different dtype
                        fill_value=1)

    def false_func():
        return paddle.full(shape=[1, 2], dtype='float32',
                        fill_value=3)

    return paddle.static.nn.cond(pred, true_func, false_func)

# 95436: sporadic failure
exportModel('conditional_block_const', test_model, [data], target_dir=sys.argv[1])


'''
more than one select_input with constant inputs.
'''
@paddle.jit.to_static
def test_model_2outputs(pred):
    # pred: A boolean tensor whose numel should be 1.
    def true_func():
        return paddle.full(shape=[1, 2], dtype='float32',
                        fill_value=1), paddle.full(shape=[1, 3], dtype='float32', # TODO: FAILED with different dtype
                        fill_value=3)

    def false_func():
        return paddle.full(shape=[3, 4], dtype='float32',
                        fill_value=3), paddle.full(shape=[1, 4], dtype='float32',
                        fill_value=4)

    return paddle.static.nn.cond(pred, true_func, false_func)

# 95436: sporadic failure
exportModel('conditional_block_const_2outputs', test_model_2outputs, [data], target_dir=sys.argv[1])


'''
more than one select_input with 2 inputs and 2 select_input nodes.
'''
@paddle.jit.to_static
def test_model_2inputs_2outputs(a, b):
    return paddle.static.nn.cond(a < b, lambda: (a, a * b), lambda: (b, a * b) )

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
# 95436: sporadic failure
exportModel('conditional_block_2inputs_2outputs', test_model_2inputs_2outputs, [a, b], target_dir=sys.argv[1])

'''
simple test case with 2 inputs to conditional_block node.
'''
@paddle.jit.to_static
def test_model2(a, b):
    c = a * b
    return paddle.static.nn.cond(a < b, lambda: a + c, lambda: b * b)

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
# 95436: sporadic failure
exportModel('conditional_block_2inputs', test_model2, [a, b], target_dir=sys.argv[1])


'''
'''
@paddle.jit.to_static
def test_model_dyn(a, b):
    c = a * b
    return a + c if a < b else b * b

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
# 95436: sporadic failure
exportModel('conditional_block_2inputs_dyn', test_model_dyn, [a, b], target_dir=sys.argv[1])

'''
more than one select_input
# looks there are bugs in paddle dygraph to static... failed to generate 2 select_inputs.
'''
@paddle.jit.to_static
def test_model_dyn_2outputs(a, b):
    c = a * b
    return a, c  if a < b else b, c

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
# 95436: sporadic failure
exportModel('conditional_block_2inputs_dyn_2outputs', test_model_dyn_2outputs, [a, b], target_dir=sys.argv[1])


""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""question: how to make test case for only one 
conditional_block ??   """
""""""""""""""""""""""""""""""""""""""""""""""""""""""
@paddle.jit.to_static
def test_model_dyn_conditionalblock_only(a):
    rpn_rois_list = []

    if a.shape[0] >= 1:
        rpn_rois_list.append(a)

    return rpn_rois_list[0]

a = paddle.to_tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
# 95436: sporadic failure
exportModel('conditional_block_dyn_conditionalblock_only', test_model_dyn_conditionalblock_only, [a], target_dir=sys.argv[1])


""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""return type: tensor, tuple(tensor), list(tensor) 
question: how to work with LoDTensorArray ??   """
""""""""""""""""""""""""""""""""""""""""""""""""""""""

# '''
# test: only conditional_block node in the pattern.
# '''
# @paddle.jit.to_static
# def test_model_return_tuple(a, b):
#     return paddle.static.nn.cond(a < b, lambda: (tuple((a, a * b))), lambda: (b, a * b) )

# a = np.full(shape=[1], dtype='float32', fill_value=0.1)
# b = np.full(shape=[1], dtype='float32', fill_value=0.23)
# exportModel('conditional_block_return_tuple', test_model_return_tuple, [a, b], target_dir=sys.argv[1])

# x = np.full(shape=[1], dtype='float32', fill_value=0.1)
# y = np.full(shape=[1], dtype='float32', fill_value=0.23)
# data = np.less(y,x)
# def test_model(pred):
#     # pred: A boolean tensor whose numel should be 1.
#     def true_func():
#         return (paddle.full(shape=[3, 4], dtype='float32', # TODO: FAILED with different dtype
#                         fill_value=1), paddle.full(shape=[3, 4], dtype='float32', # TODO: FAILED with different dtype
#                         fill_value=1))

#     def false_func():
#         return (paddle.full(shape=[1, 2], dtype='float32',
#                         fill_value=3),paddle.full(shape=[1, 2], dtype='float32',
#                         fill_value=3))

#     return paddle.static.nn.cond(pred, true_func, false_func)

# exportModel('conditional_block_return_tuple2', test_model, [data], target_dir=sys.argv[1])


""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""stack blocks"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""

'''
select_output with multiple consumers.
'''
@paddle.jit.to_static
def test_model_dyn_multiple_consumers(a, b):
    c = a * b
    cond0 = a + c if a < b else b * b
    o1 = cond0 + a
    o2 = cond0 + b
    return o1, o2

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
# 95436: sporadic failure
exportModel('conditional_block_dyn_multiple_consumers', test_model_dyn_multiple_consumers, [a, b], target_dir=sys.argv[1])

'''
stack if-else blocks
'''
@paddle.jit.to_static
def test_model_dyn_multiple_blocks(a, b, c):
    cond0 = a + c if a < b else b * b
    cond1 = a - c if b < c else b * b

    return cond0, cond1

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
c = np.full(shape=[1], dtype='float32', fill_value=0.9)
# 95436: sporadic failure
exportModel('conditional_block_dyn_multiple_blocks', test_model_dyn_multiple_blocks, [a, b, c], target_dir=sys.argv[1])

@paddle.jit.to_static
def test_model_dyn_multiple_blocks2(a, b, c):
    cond0 = a + c if a < b else b * b
    cond1 = a - c if cond0 < c else b * b

    return cond0, cond1

a = np.full(shape=[1], dtype='float32', fill_value=0.1)
b = np.full(shape=[1], dtype='float32', fill_value=0.23)
c = np.full(shape=[1], dtype='float32', fill_value=0.9)
# 95436: sporadic failure
exportModel('conditional_block_dyn_multiple_blocks2', test_model_dyn_multiple_blocks2, [a, b, c], target_dir=sys.argv[1])


