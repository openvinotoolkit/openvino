# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# strided_slice paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def strided_slice(name: str, input_data, attrs: dict):
    import paddle

    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        Input = paddle.static.data(
            name="x", shape=input_data.shape, dtype=input_data.dtype
        )

        if paddle.__version__ >= '2.0.0':
            out = paddle.strided_slice(
                Input,
                axes=attrs["axes"],
                starts=attrs["starts"],
                ends=attrs["ends"],
                strides=attrs["strides"],
            )
        else:
            out = paddle.fluid.layers.strided_slice(Input, axes=attrs['axes'],
                                                    starts=attrs['starts'],
                                                    ends=attrs['ends'],
                                                    strides=attrs['strides'])

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": input_data}, fetch_list=[out])

        # Save inputs in order of OpenVINO model, to facilite Fuzzy test,
        # which accepts inputs and outputs in this order as well.
        saveModel(
            name,
            exe,
            feed_vars=[Input],
            fetchlist=[out],
            inputs=[input_data],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )
    return outs


if __name__ == "__main__":
    strided_slice_input1_1 = {
        "name": "strided_slice_input1_1",
        "axes": np.array([0]).astype("int32").tolist(),
        "starts": np.array([-4]).astype("int32").tolist(),
        "ends": np.array([-3]).astype("int32").tolist(),
        "strides": np.array([1]).astype("int32").tolist(),
    }

    strided_slice_input1_2 = {
        "name": "strided_slice_input1_2",
        "axes": np.array([0]).astype("int32").tolist(),
        "starts": np.array([3]).astype("int32").tolist(),
        "ends": np.array([8]).astype("int32").tolist(),
        "strides": np.array([1]).astype("int32").tolist(),
    }

    strided_slice_input1_3 = {
        "name": "strided_slice_input1_3",
        "axes": np.array([0]).astype("int32").tolist(),
        "starts": np.array([5]).astype("int32").tolist(),
        "ends": np.array([0]).astype("int32").tolist(),
        "strides": np.array([-1]).astype("int32").tolist(),
    }

    strided_slice_input1_4 = {
        "name": "strided_slice_input1_4",
        "axes": np.array([0]).astype("int32").tolist(),
        "starts": np.array([-1]).astype("int32").tolist(),
        "ends": np.array([-3]).astype("int32").tolist(),
        "strides": np.array([-1]).astype("int32").tolist(),
    }

    strided_slice_input2_1 = {
        "name": "strided_slice_input2_1",
        "axes": np.array([0, 1, 2]).astype("int32").tolist(),
        "starts": np.array([1, 0, 0]).astype("int32").tolist(),
        "ends": np.array([2, 1, 3]).astype("int32").tolist(),
        "strides": np.array([1, 1, 1]).astype("int32").tolist(),
    }

    strided_slice_input2_2 = {
        "name": "strided_slice_input2_2",
        "axes": np.array([0, 1, 2]).astype("int32").tolist(),
        "starts": np.array([1, -1, 0]).astype("int32").tolist(),
        "ends": np.array([2, -3, 3]).astype("int32").tolist(),
        "strides": np.array([1, -1, 1]).astype("int32").tolist(),
    }

    strided_slice_input2_3 = {
        "name": "strided_slice_input2_3",
        "axes": np.array([0, 1, 2]).astype("int32").tolist(),
        "starts": np.array([1, 0, 0]).astype("int32").tolist(),
        "ends": np.array([2, 2, 3]).astype("int32").tolist(),
        "strides": np.array([1, 1, 1]).astype("int32").tolist(),
    }

    strided_slice_input3_1 = {
        "name": "strided_slice_input3_1",
        "axes": np.array([1]).astype("int32").tolist(),
        "starts": np.array([1]).astype("int32").tolist(),
        "ends": np.array([2]).astype("int32").tolist(),
        "strides": np.array([1]).astype("int32").tolist(),
    }

    strided_slice_input3_2 = {
        "name": "strided_slice_input3_2",
        "axes": np.array([1]).astype("int32").tolist(),
        "starts": np.array([-1]).astype("int32").tolist(),
        "ends": np.array([-2]).astype("int32").tolist(),
        "strides": np.array([-1]).astype("int32").tolist(),
    }

    strided_slice_input1_list = [
        strided_slice_input1_1,
        strided_slice_input1_2,
        strided_slice_input1_3,
        strided_slice_input1_4,
    ]

    strided_slice_input2_list = [
        strided_slice_input2_1,
        strided_slice_input2_2,
        strided_slice_input2_3,
    ]

    strided_slice_input3_list = [strided_slice_input3_1, strided_slice_input3_2]

    input1 = np.random.rand(100).astype("float32")
    for item in strided_slice_input1_list:
        pred_paddle = strided_slice(item["name"], input1, item)

    input2 = np.random.rand(5, 5, 5).astype("int32")
    for item in strided_slice_input2_list:
        pred_paddle = strided_slice(item["name"], input2, item)

    input3 = np.random.rand(1, 100, 1).astype("float32")
    for item in strided_slice_input3_list:
        pred_paddle = strided_slice(item["name"], input3, item)
