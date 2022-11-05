# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# reverse paddle model generator
#
import numpy as np
from save_model import saveModel
import sys
import paddle

def reverse(name: str, x, axis, use_static=True, dtype="float32"):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        if use_static:
            node_x = paddle.static.data(name='x', shape=x.shape, dtype=dtype)
        else:
            node_x = paddle.fluid.data(name='x', shape=[1, 1, -1, -1], dtype=dtype)
        out = paddle.fluid.layers.reverse(node_x, axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    data1 = np.array([0,2], dtype='int64')
    reverse("reverse_static_1", data1, 0, True, 'int64')

    data2 = np.array([[0,1,2], [3,4,5], [6,7,8]], dtype='float32')
    reverse("reverse_static_2", data2, 1, True, 'float32')

    data3 = np.array([[0,1,2], [3,4,5], [6,7,8]], dtype='float32')
    reverse("reverse_static_3", data3, [0, 1], True, 'float32')

    data4 = np.array([[0,1,2], [3,4,5], [6,7,8]], dtype='int32')
    reverse("reverse_static_4", data4, -1, True, 'int32')

    data5 = np.random.randn(1, 1, 32, 32).astype('int32')
    reverse("reverse_dynamic_1", data5, [2], False, dtype='int32')

    data6 = np.random.randn(1, 1, 64, 64).astype('float32')
    reverse("reverse_dynamic_2", data6, [3], False, dtype='float32')

    data7 = np.random.randn(1, 1, 112, 112).astype('float32')
    reverse("reverse_dynamic_3", data7, [2,3], False, dtype='float32')

    data8 = np.random.randn(1, 1, 224, 224).astype('int32')
    reverse("reverse_dynamic_4", data8, [-2, -1], False, dtype='int32')

if __name__ == "__main__":
    main()
