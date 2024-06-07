# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# full_like paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'

def full_like(name:str, x, value, dtype=None):
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name='x', shape=x.shape, dtype = x.dtype)
        out = paddle.full_like(data, value, dtype=dtype)
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
    x = np.random.rand(8, 24, 32).astype(data_type)

    full_like("full_like", x, 1.2)
    full_like("full_like_f16", x, 1.0, dtype='float16')
    full_like("full_like_f32", x, 1.2, dtype='float32')
    full_like("full_like_f64", x, 1.2, dtype='float64')
    full_like("full_like_i16", x, 3, dtype='int16')
    full_like("full_like_i32", x, 2, dtype='int32')
    full_like("full_like_i64", x, 10, dtype='int64')
    full_like("full_like_bool", x, True, dtype='bool')

    sample_arr = [True, False]
    x = np.random.choice(sample_arr, size=(13,17,11))
    full_like("full_like_bool_2", x, False, dtype=None)

if __name__ == "__main__":
    main()
