# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# fill_any_like paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'

def fill_any_like(name:str, x, value, dtype=None):
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name='x', shape=x.shape, dtype = data_type)
        out = paddle.full_like(data, value, dtype=dtype)
        out = paddle.cast(out, np.float32)

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
    x = np.random.rand(8, 24, 32).astype(data_type)

    fill_any_like("fill_any_like", x, 1.2)
    fill_any_like("fill_any_like_f16", x, 1.0, dtype='float16')
    fill_any_like("fill_any_like_f32", x, 1.2, dtype='float32')
    fill_any_like("fill_any_like_f64", x, 1.2, dtype='float64')
    fill_any_like("fill_any_like_i32", x, 2, dtype='int32')
    fill_any_like("fill_any_like_i64", x, 10, dtype='int64')

if __name__ == "__main__":
    main()