# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# one_hot_v2 paddle model generator
#
import numpy as np
from save_model import saveModel
import sys

def one_hot_v2(name: str, x, num_classes):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = paddle.nn.functional.one_hot(node_x, num_classes)

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
    classes = 3
    data = np.random.randint(0, classes, [4]).astype('int32')
    one_hot_v2("one_hot_v2_int32", data, classes)

    classes = 20
    data = np.random.randint(0, classes, [100]).astype('int64')
    one_hot_v2("one_hot_v2_int64", data, classes)

if __name__ == "__main__":
    main()
