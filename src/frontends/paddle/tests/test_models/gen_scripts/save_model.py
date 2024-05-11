# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import paddle

#print numpy array like C structure       
def print_alike(arr, seperator_begin='{', seperator_end='}', verbose=False):
    shape = arr.shape
    rank = len(shape)

    #print("shape: ", shape, "rank: %d" %(rank))

    # for idx, value in np.ndenumerate(arr):
    #    print(idx, value)

    def print_array(arr, end=' '):
        shape = arr.shape
        rank = len(arr.shape)
        if rank > 1:
            line = seperator_begin
            for i in range(arr.shape[0]):
                line += print_array(
                    arr[i, :],
                    end=seperator_end +
                    ",\n" if i < arr.shape[0] - 1 else seperator_end)
            line += end
            return line
        else:
            line = seperator_begin
            for i in range(arr.shape[0]):
                line += "{:.2f}".format(arr[i])  # str(arr[i])
                line += ", " if i < shape[0] - 1 else ' '
            line += end
            # print(line)
            return line

    if verbose:
        print(print_array(arr, seperator_end))        

def saveModel(name, exe, feed_vars:list, fetchlist:list, inputs:list, outputs:list, target_dir:str):
    model_dir = os.path.join(target_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # print("\n\n------------- %s -----------\n" % (name))
    for i, input in enumerate(inputs):
        feedkey = feed_vars[i].name
        # print("INPUT %s :" % (feedkey), input.shape, input.dtype, "\n")
        # print_alike(input)
        np.save(os.path.join(model_dir, "input{}".format(i)), input)
        np.save(os.path.join(model_dir, "input{}.{}.{}".format(i, feedkey, input.dtype)), input)
    # print("\n")

    for i, output in enumerate(outputs):
        # print("OUTPUT %s :" % (fetchlist[i]),output.shape, output.dtype, "\n")
        # print_alike(output)
        np.save(os.path.join(model_dir, "output{}".format(i)), output)

    # composited model + scattered model
    model_name = os.path.join(model_dir, name)
    paddle.static.io.save_inference_model(model_name, feed_vars, fetchlist, exe)


'''
export dyn model, along with input and output for reference.
input_data: list of all inputs
'''
def exportModel(name, dyn_func, input_data:list, target_dir:str, dyn_shapes:list=[]):
    model_dir = os.path.join(target_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_path = '{}/{}'.format(model_dir, name)

    if len(dyn_shapes)>0:
        assert(len(dyn_shapes) == len(input_data))

    input_specs = []
    for idx, data in enumerate(input_data):
        input_name = 'input{}'.format(idx)
        input_shape = dyn_shapes[idx] if len(dyn_shapes)>0 and dyn_shapes[idx] is not None else data.shape

        input_specs.append(
            paddle.static.InputSpec(shape=input_shape, dtype=data.dtype, name=input_name)
        )

        # dump input
        np.save(os.path.join(model_dir, "input{}".format(idx)), data)        

    paddle.jit.save(dyn_func, save_path, input_specs)
    print('saved exported model to {}'.format(save_path))

    # infer
    model = paddle.jit.load(save_path)

    result = model(*[input[:] for input in input_data])
   
    # dump output for reference
    if isinstance(result, (tuple, list)):
        for idx, out in enumerate(result):
            np.save(os.path.join(model_dir, "output{}".format(idx)), out.numpy())
    else:       
        np.save(os.path.join(model_dir, "output{}".format(0)), result.numpy())
    
    if paddle.__version__ < "2.6.0": 
        paddle.fluid.core.clear_executor_cache()
    else:
        paddle.base.core.clear_executor_cache()
    return result


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)  

    #x = np.random.randn(2,3).astype(np.float32)
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [1, 2, 3],
        [4, 5, 6]
    ]], 
    [[
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [1, 2, 3],
        [4, 5, 6]
    ]]]).astype(np.float32)
