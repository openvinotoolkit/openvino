from __future__ import print_function
import sys
import numpy as np
import paddle

from save_model import saveModel


def sum_(name: str, input):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data('data', shape=input.shape, dtype=input.dtype)
        out = paddle.add_n(data)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={"data": input},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[data], fetchlist=[out], inputs=[input], outputs=[outs[0]],
                  target_dir=sys.argv[1])
    return outs[0]


def sum(name: str, inputs):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data0 = paddle.static.data('data0', shape=inputs[0].shape, dtype=inputs[0].dtype)
        data1 = paddle.static.data('data1', shape=inputs[1].shape, dtype=inputs[1].dtype)
        data2 = paddle.static.data('data2', shape=inputs[2].shape, dtype=inputs[2].dtype)
        out = paddle.add_n([data0, data1, data2])
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={"data0": inputs[0],
                  "data1": inputs[1],
                  "data2": inputs[2]},
            fetch_list=[out])
        saveModel(name, exe, feed_vars=[data0, data1, data2], fetchlist=[out],
                  inputs=[inputs[0], inputs[1], inputs[2]], outputs=[outs[0]],
                  target_dir=sys.argv[1])
    return outs[0]


def main():
    # single tensor
    input = np.random.random([2, 3]).astype(np.float32)
    sum_("sum_1", input)
    # multiple tensors with type float32
    input1 = np.random.random([2, 3]).astype(np.float32)
    input2 = np.random.random([2, 3]).astype(np.float32)
    input3 = np.random.random([2, 3]).astype(np.float32)
    sum("sum_2", [input1, input2, input3])
    # multiple tensors with type int32
    input1 = np.random.random([2, 3]).astype(np.int32)
    input2 = np.random.random([2, 3]).astype(np.int32)
    input3 = np.random.random([2, 3]).astype(np.int32)
    sum("sum_3", [input1, input2, input3])
    # multiple tensors with type int64
    input1 = np.random.random([2, 3]).astype(np.int64)
    input2 = np.random.random([2, 3]).astype(np.int64)
    input3 = np.random.random([2, 3]).astype(np.int64)
    sum("sum_4", [input1, input2, input3])


if __name__ == "__main__":
    main()
