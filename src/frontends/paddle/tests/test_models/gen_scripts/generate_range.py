# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# range paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def paddle_range(name : str, x, start, end, step, out_type):
    import paddle
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        # Range op only support fill_constant input, since dynamic op is not supported in ov
        if paddle.__version__ >= '2.0.0':
            out = paddle.arange(start, end, step, out_type)
        else:
            out = paddle.fluid.layers.range(start, end, step, out_type)
        out = paddle.cast(out, np.float32)
        out = paddle.add(node_x, out)
        #out = paddle.cast(out, np.float32)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
                feed={'x': x},
                fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    start = 1.5
    end = 10.5
    step = 2
    data = np.random.random([1, 5]).astype("float32")
    out_type = ["float32", "int32", "int64"]
    for i, dtype in enumerate(out_type):
        paddle_range("range"+str(i), data, start, end, step, dtype)


if __name__ == "__main__":
    main()     
