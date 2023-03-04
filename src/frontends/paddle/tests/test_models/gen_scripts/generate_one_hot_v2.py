# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# one_hot_v2 paddle model generator
#
import paddle
import numpy as np
from save_model import saveModel
import sys


def one_hot_v2_1(name: str, x, num_classes):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_node = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        out = paddle.nn.functional.one_hot(x_node, num_classes=num_classes)
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        outs = exe.run(feed={"x": x}, fetch_list=[out])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def one_hot_v2_2(name: str, x, num_classes):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_node = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        depth_node = paddle.static.data(name="depth_tensor", shape=num_classes.shape, dtype=num_classes.dtype)
        out = paddle.nn.functional.one_hot(x_node, num_classes=depth_node)
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        outs = exe.run(feed={"x": x, "depth_tensor": num_classes}, fetch_list=[out])
        saveModel(name, exe, feedkeys=["x", "depth_tensor"], fetchlist=[out], inputs=[x, num_classes], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    # int32
    data = np.array([1, 1, 3, 0]).astype("int32")
    num_classes = 4
    one_hot_v2_1("one_hot_v2_1", data, num_classes)
    # int64
    data = np.array([4, 1, 3, 3]).astype("int64")
    num_classes = np.array([5]).astype("int32")
    one_hot_v2_2("one_hot_v2_2", data, num_classes)


if __name__ == "__main__":
    main()
