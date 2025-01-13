# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# one_hot_v2 paddle model generator
#
import paddle
import numpy as np
from save_model import saveModel
import sys


def one_hot_v2(name: str, x, num_classes, is_tensor):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_node = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        depth_node = paddle.static.data(name="depth_tensor", shape=num_classes.shape, dtype=num_classes.dtype) if is_tensor else num_classes
        out = paddle.nn.functional.one_hot(x_node, num_classes=depth_node)
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        feed_list = {"x": x, "depth_tensor": num_classes} if is_tensor else {"x": x}
        outs = exe.run(feed=feed_list, fetch_list=[out])
        feed_vars = [x_node, depth_node] if is_tensor else [x_node]
        input_list = [x, num_classes] if is_tensor else [x]
        saveModel(name, exe, feed_vars=feed_vars, fetchlist=[out], inputs=input_list, outputs=[outs[0]], target_dir=sys.argv[1])
    
    return outs[0]


def main():
    # int 32
    data = np.array([1]).astype("int32")
    num_classes = 4
    one_hot_v2("one_hot_v2_1", data, num_classes, is_tensor=False)
    # rank 1 int64
    data = np.array([4, 1, 3, 3]).astype("int64")
    num_classes = np.array([5]).astype("int32")
    one_hot_v2("one_hot_v2_2", data, num_classes, is_tensor=True)
    # rank 2 int64
    data = np.array([[4, 1, 3, 3], [1, 1, 3, 0]]).astype("int64")
    num_classes = np.array([5]).astype("int32")
    one_hot_v2("one_hot_v2_3", data, num_classes, is_tensor=True)


if __name__ == "__main__":
    main()
