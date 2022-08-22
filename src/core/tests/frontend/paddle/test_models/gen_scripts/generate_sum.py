# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# stack paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def sum(name:str, input1, input2, input3):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data1 = paddle.static.data(
            'data1', shape=input1.shape, dtype=input1.dtype)
        data2 = paddle.static.data(
            'data2', shape=input2.shape, dtype=input2.dtype)
        data3 = paddle.static.data(
            'data3', shape=input3.shape, dtype=input3.dtype)

        out = paddle.fluid.layers.sum([data1, data2, data3])
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={"data1": input1,
                  "data2": input2,
                  "data3": input3},
            fetch_list=[out])
        saveModel(name, exe, feedkeys=['data1', 'data2', 'data3'], fetchlist=[out], inputs=[
                input1, input2, input3], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    
    in_type = np.float32
    in_shape = [1, 5]
    input1 = np.random.random(in_shape).astype(in_type)
    input2 = np.random.random(in_shape).astype(in_type)
    input3 = np.random.random(in_shape).astype(in_type)
    sum("sum_float_1", input1, input2, input3)

    in_type = np.float32
    in_shape = [5, 5]
    input1 = np.random.random(in_shape).astype(in_type)
    input2 = np.random.random(in_shape).astype(in_type)
    input3 = np.random.random(in_shape).astype(in_type)
    sum("sum_float_2", input1, input2, input3)

    in_type = np.float32
    in_shape = [5, 10]
    input1 = np.random.random(in_shape).astype(in_type)
    input2 = np.random.random(in_shape).astype(in_type)
    input3 = np.random.random(in_shape).astype(in_type)
    sum("sum_float_3", input1, input2, input3)


if __name__ == "__main__":
    main()
