# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# stack paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def stack(axis, input1, input2, input3):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data1 = paddle.static.data(
            'data1', shape=input1.shape, dtype=input1.dtype)
        data2 = paddle.static.data(
            'data2', shape=input2.shape, dtype=input2.dtype)
        data3 = paddle.static.data(
            'data3', shape=input3.shape, dtype=input3.dtype)

        if (axis == None):
            out = paddle.paddle.stack([data1, data2, data3])
        else:
            out = paddle.paddle.stack([data1, data2, data3], axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={"data1": input1,
                  "data2": input2,
                  "data3": input3},
            fetch_list=[out])

        if (axis == None):
            saveModel("stack_test_none_axis", exe, feed_vars=[data1, data2, data3], fetchlist=[out], inputs=[
                input1, input2, input3], outputs=[outs[0]], target_dir=sys.argv[1])
        elif (axis < 0):
            saveModel("stack_test_neg_axis", exe, feed_vars=[data1, data2, data3], fetchlist=[out], inputs=[
                input1, input2, input3], outputs=[outs[0]], target_dir=sys.argv[1])
        else:
            saveModel("stack_test_" + str(input1.dtype), exe, feed_vars=[data1, data2, data3], fetchlist=[out], inputs=[
                input1, input2, input3], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    in_dtype = np.float32
    axis_num = 1
    input1 = np.random.random([1, 2]).astype(in_dtype)
    input2 = np.random.random([1, 2]).astype(in_dtype)
    input3 = np.random.random([1, 2]).astype(in_dtype)
    stack(axis_num, input1, input2, input3)

    in_dtype = np.int32
    axis_num = 0
    input1 = np.random.random([1, 2]).astype(in_dtype)
    input2 = np.random.random([1, 2]).astype(in_dtype)
    input3 = np.random.random([1, 2]).astype(in_dtype)
    stack(axis_num, input1, input2, input3)

    in_dtype = np.float32
    axis_num = None
    input1 = np.random.random([1, 2]).astype(in_dtype)
    input2 = np.random.random([1, 2]).astype(in_dtype)
    input3 = np.random.random([1, 2]).astype(in_dtype)
    stack(axis_num, input1, input2, input3)

    in_dtype = np.float32
    axis_num = -1
    input1 = np.random.random([1, 2]).astype(in_dtype)
    input2 = np.random.random([1, 2]).astype(in_dtype)
    input3 = np.random.random([1, 2]).astype(in_dtype)
    stack(axis_num, input1, input2, input3)


if __name__ == "__main__":
    main()
