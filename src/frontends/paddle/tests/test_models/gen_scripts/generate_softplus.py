# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# softplus paddle model generator
#
import numpy as np
import sys
from save_model import saveModel


def softplus(name: str, x, beta, threshold):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = paddle.nn.functional.softplus(x=node_x, beta=beta, threshold=threshold)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        if x.dtype == np.float16:
            # Paddle CPU has no float16 Softplus kernel. Serialize the f16 model
            # without executing it and calculate the reference output with NumPy.
            x_f32 = x.astype(np.float32)
            scaled = beta * x_f32
            reference = np.where(
                scaled > threshold,
                x_f32,
                np.log1p(np.exp(np.minimum(scaled, threshold))) / beta,
            )
            outs = [reference.astype(x.dtype)]
        else:
            outs = exe.run(
                feed={'x': x},
                fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.array(
        [[[2.0, 3.0, 4.0, 5.0],
          [3.0, 4.0, 5.0, 6.0],
          [7.0, 8.0, 8.0, 9.0]],
         [[1.0, 2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0, 8.0],
          [6.0, 7.0, 8.0, 9.0]]]
    ).astype(np.float32)

    softplus("softplus_default_params", data, beta=1, threshold=20)
    # issue 37951: non-default beta/threshold must convert (previously rejected); the input
    # crosses the saturation point beta*x = threshold -> x = 15
    x = np.array([-10.0, -1.0, 0.0, 1.0, 2.0, 8.0, 14.9, 15.1]).astype(np.float32)
    softplus("softplus_beta2_threshold30", x, beta=2, threshold=30)
    # A low threshold pins the saturated branch: beta*x = threshold -> x = 1, and next to that point
    # the linear branch differs from the formula one by ~0.06-0.08, i.e. far above the fuzzy
    # tolerance, so ignoring or hard-coding the threshold cannot pass.
    x_low_threshold = np.array([-2.0, -0.5, 0.5, 0.9, 1.0, 1.1, 2.0, 3.0]).astype(np.float32)
    softplus("softplus_beta2_threshold2", x_low_threshold, beta=2, threshold=2)
    # f16 regression: 1 / beta overflows f16, but softplus(0) / beta is finite.
    x_f16 = np.array([0.0], dtype=np.float16)
    softplus("softplus_f16_small_beta", x_f16, beta=2.0**-16, threshold=20)

if __name__ == "__main__":
    main()
