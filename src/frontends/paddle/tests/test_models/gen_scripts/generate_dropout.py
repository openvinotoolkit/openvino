# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def paddle_dropout(name : str, x, p, paddle_attrs):
    import paddle
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        out = paddle.nn.functional.dropout(x=node_x, p=p, training=paddle_attrs['training'], mode=paddle_attrs['mode'])

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x],
                  outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    p=0.5
    data = np.random.random(size=(3, 10, 3, 7)).astype('float32')
    paddle_attrs = {
    'training' : False,
    'mode' : "downscale_in_infer"
    }
    paddle_attrs2 = {
        'training' : False,
        'mode' : "upscale_in_train"
    }
    paddle_dropout("dropout", data, p, paddle_attrs)
    paddle_dropout("dropout_upscale_in_train", data, p, paddle_attrs2)

if __name__ == "__main__":
    main()     
