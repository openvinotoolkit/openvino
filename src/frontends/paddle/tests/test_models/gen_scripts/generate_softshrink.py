# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# clip paddle model generator
#
import numpy as np
from save_model import saveModel
import sys

def softshrink(name: str, x, threshold=0.5):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        softshrink_f = paddle.nn.Softshrink(threshold)
        out = softshrink_f(node_x)
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
    data = np.random.random([2, 3, 4]).astype('float32')
    softshrink("softshrink_default_attribute", data)
    softshrink("softshrink", data, 0.1)



if __name__ == "__main__":
    main()
