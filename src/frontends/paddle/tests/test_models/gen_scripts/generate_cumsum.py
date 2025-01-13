# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# cumsum paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'

def cumsum(name:str, x, axis, dtype=None):
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name='x', shape=x.shape, dtype = data_type)
        out = paddle.cumsum(data, axis, dtype=dtype)
        out = paddle.cast(out, np.float32)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[data], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    x = np.linspace(1, 12, 12, dtype=data_type)
    x = np.reshape(x, (3, 4))

    cumsum("cumsum", x, axis=None)
    cumsum("cumsum_f32", x, axis=-1, dtype='float32')
    cumsum("cumsum_f64", x, axis=0, dtype='float64')
    cumsum("cumsum_i32", x, axis=0, dtype='int32')
    cumsum("cumsum_i64", x, axis=0, dtype='int64')

if __name__ == "__main__":
    main()
