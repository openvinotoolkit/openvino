import os
import sys

import numpy as np
import paddle

from save_model import exportModel

""""""""""""""""""""""""""""""""""""""""""""""""""""""
# tensorarray case: conditional_block + slice[0]
""""""""""""""""""""""""""""""""""""""""""""""""""""""
def test_conditional_block_slice0():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)

                return rpn_rois_list[0]    
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    test('conditional_block_slice0', [a])
    a_shape = a.shape
    a_shape[0] = -1
    test('conditional_block_slice0_dyn', [a], [a_shape])

def test_conditional_block_slice0_else():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)
                else:
                    rpn_rois_list.append(a)

                return rpn_rois_list[0]
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    test('conditional_block_slice0_else', [a])

# wrong test case. paddle throw error " IndexError: list index out of range" in paddle.jit.save()
def test_conditional_block_slice0_empty():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                return rpn_rois_list[0]    
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    test('conditional_block_slice0_empty', [a])

# wrong test case. paddle throw error " IndexError: list index out of range" in paddle.jit.save()
def test_conditional_block_concat_empty():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                return paddle.concat(rpn_rois_list)
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    test('conditional_block_concat_empty', [a])    

# could generate paddle model, but paddle throw exception during inferencing.
def test_conditional_block_slice0_empty_false():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                if a.shape[0] >= 1000: # false condition
                    rpn_rois_list.append(a)

                return rpn_rois_list[0]
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    test('conditional_block_slice0_empty_false', [a])    

def test_conditional_block_slice0_scaler():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)

                return rpn_rois_list[0]
            return exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor(10.0)
    print(test('conditional_block_slice0_scaler', [a]))
    a_shape = a.shape
    a_shape[0] = -1
    print(test('conditional_block_slice0_scaler_dyn', [a], [a_shape]))

# No such case in faster/mask rcnn... as paddle.concat always concat along axis 0.
def test_conditional_block_slice0_axis2():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)

                return rpn_rois_list[0]
            return exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.arange(2*3*4*5)
    a = paddle.reshape(a, (2, 3, 4, 5)).astype('float32')
    print(test('conditional_block_slice0_axis2', [a]))
    a_shape = a.shape
    a_shape[2] = -1 # instead of 0
    print(test('conditional_block_slice0_axis2_dyn', [a], [a_shape]))

# No such case in faster/mask rcnn... as paddle.concat always concat along axis 0.
def test_conditional_block_slice0_aixs1_axis2():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)

                return rpn_rois_list[0]
            return exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.arange(2,2,3)
    a = paddle.reshape(a, (2,2,3)).astype('float32')
    print(test('conditional_block_slice0_axis1_axis2', [a]))
    a_shape = a.shape
    a_shape[1] = -1 # instead of 0
    a_shape[2] = -1 # instead of 0
    print(test('conditional_block_slice0_axis1_axis2_dyn', [a], [a_shape]))

def test_conditional_block_slice1():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a, b):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)
                    rpn_rois_list.append(b)

                return rpn_rois_list[1]
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    b = paddle.to_tensor( [[7.0, 8.0, 9.0]])    
    test('conditional_block_slice1', [a, b])
    a_shape = a.shape
    a_shape[0] = -1
    test('conditional_block_slice1_dyn', [a, b], [a_shape, b.shape])

def test_conditional_block_slice0_slice1():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a, b):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)
                    rpn_rois_list.append(b)

                return rpn_rois_list[0], rpn_rois_list[1]
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    b = paddle.to_tensor( [[7.0, 8.0, 9.0]])                
    test('conditional_block_slice0_slice1', [a, b])
    a_shape = a.shape
    a_shape[0] = -1
    test('conditional_block_slice0_slice1_dyn', [a, b], [a_shape, b.shape])

def test_conditional_block_slice0_2tensorarrays():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a, b):
                rpn_rois_list = []
                rpn_rois_list1 = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)
                    rpn_rois_list1.append(b)

                return rpn_rois_list[0], rpn_rois_list1[0]
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    b = paddle.to_tensor( [[7.0, 8.0, 9.0]])
    test('conditional_block_slice0_2tensorarrays', [a, b])
    a_shape = a.shape
    a_shape[0] = -1
    b_shape = b.shape
    b_shape[0] = -1
    test('conditional_block_slice0_2tensorarrays_dyn', [a, b], [a_shape, b_shape])

def test_conditional_block_slice0_2tensorarrays_extra():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a, b):
                rpn_rois_list = []
                rpn_rois_list1 = []

                if a.shape[0] >= 1:
                    c = a + b
                    rpn_rois_list.append(a)
                    rpn_rois_list.append(c)
                    rpn_rois_list1.append(b)

                return paddle.concat(rpn_rois_list), rpn_rois_list1[0]
            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    b = paddle.to_tensor( [[7.0, 8.0, 9.0]])
    test('conditional_block_slice0_2tensorarrays_extra', [a, b])
    a_shape = a.shape
    a_shape[0] = -1
    b_shape = b.shape
    b_shape[0] = -1
    test('conditional_block_slice0_2tensorarrays_extra_dyn', [a, b], [a_shape, b_shape])

def test_conditional_block_concat():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a, b):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)
                    rpn_rois_list.append(b)

                return paddle.concat(rpn_rois_list)

            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]]])
    b = paddle.to_tensor([[[7.0, 8.0, 9.0],
                          [10.0, 11.0, 12.0]]])
    test('conditional_block_concat', [a, b])

    a_shape = a.shape
    a_shape[1] = -1
    test('conditional_block_concat_dyn', [a, b], [a_shape, b.shape])

    # the case of mask_rcnn_r50_1x_coco block13.
    a_shape = a.shape
    a_shape[1] = -1
    a_shape[2] = -1
    test('conditional_block_concat_dyn_2axes', [a, b], [a_shape, b.shape])

# the condition is false
def test_conditional_block_concat_false():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a, b):
                rpn_rois_list = []

                if a.shape[0] >= 1:
                    rpn_rois_list.append(a)

                if a.shape[0] >= 100: # False condition
                    rpn_rois_list.append(b)

                return paddle.concat(rpn_rois_list)

            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    b = paddle.to_tensor([[7.0, 8.0, 9.0],
                          [10.0, 11.0, 12.0]])
    test('conditional_block_concat_false', [a, b])

    a_shape = a.shape
    a_shape[0] = -1
    test('conditional_block_concat_false_dyn', [a, b], [a_shape, b.shape])

"""
conditional_block connects to another conditional_block.
This is the case of faster/mask rcnn with fpn.
"""
def test_conditional_block_conditional_block_concat():
    def test(model_name, inputs:list, input_shapes=[]):
        with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
            @paddle.jit.to_static
            def test_model(a):
                rpn_rois_list = []

                for i in range(5):
                    if a.shape[0] >= 1:
                        rpn_rois_list.append(a)

                return paddle.concat(rpn_rois_list)

            exportModel(model_name, test_model, inputs, target_dir=sys.argv[1], dyn_shapes=input_shapes)

    a = paddle.to_tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
    test('conditional_block_conditional_block_concat', [a])
    a_shape = a.shape
    a_shape[0] = -1
    test('conditional_block_conditional_block_concat_dyn', [a], [a_shape])

if __name__ == "__main__":
    # Failure if update paddlepaddle to 2.4.0
    #test_conditional_block_slice0()
    #test_conditional_block_slice0_scaler()
    #test_conditional_block_concat()
    #test_conditional_block_concat_false()
    #test_conditional_block_conditional_block_concat()
    #test_conditional_block_slice0_axis2()
    #test_conditional_block_slice0_2tensorarrays()
    #test_conditional_block_slice0_2tensorarrays_extra()
    #test_conditional_block_slice0_else()
    ## 95436: sporadic failure
    ##test_conditional_block_slice1()
    ##test_conditional_block_slice0_slice1()
    ##test_conditional_block_slice0_aixs1_axis2()
    ## test_conditional_block_slice0_empty()
    ## test_conditional_block_concat_empty()
    ## test_conditional_block_slice0_empty_false()
    pass
