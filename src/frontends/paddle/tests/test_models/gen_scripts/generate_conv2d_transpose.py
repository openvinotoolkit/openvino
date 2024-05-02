# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import paddle

paddle.enable_static()
import sys

from save_model import saveModel


def run_and_save_model(input_x, name, feed, fetch_list, main_prog, start_prog):
    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    exe.run(start_prog)
    outs = exe.run(feed={"x": input_x}, fetch_list=fetch_list, program=main_prog)
    with paddle.static.program_guard(main_prog, start_prog):
        saveModel(
            name,
            exe,
            feedkeys=[feed],
            fetchlist=fetch_list,
            inputs=[input_x],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
            use_static_api=True,
        )


def paddle_conv2d_transpose(
    input_x,
    name,
    input_shape,
    kernel,
    dilation,
    padding,
    stride,
    groups=1,
    use_cudnn=True,
):
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        data = paddle.static.data(name="x", shape=input_shape, dtype="float32")
        weight_attr = paddle.ParamAttr(
            name="conv2d_weight", initializer=paddle.nn.initializer.Assign(kernel)
        )
        conv2d = paddle.static.nn.conv2d_transpose(
            input=data,
            num_filters=kernel.shape[0],
            filter_size=kernel.shape[2:4],
            padding=padding,
            param_attr=weight_attr,
            dilation=dilation,
            stride=stride,
            groups=groups,
            use_cudnn=use_cudnn,
        )
    run_and_save_model(input_x, name, data, conv2d, main_program, startup_program)


if __name__ == "__main__":

    test_cases = [
        {
            "input_x": np.array(
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 7, 5) input tensor
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0, 13.0, 14.0],
                            [15.0, 16.0, 17.0, 18.0, 19.0],
                            [20.0, 21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0, 29.0],
                            [
                                30.0,
                                31.0,
                                32.0,
                                33.0,
                                34.0,
                            ],
                        ]
                    ]
                ]
            ).astype(np.float32),
            "name": "conv2d_transpose_SAME_padding",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array(
                [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
            ).astype(np.float32),
            "dilation": 1,
            "padding": "SAME",
            "stride": 2,
        },
        {
            "input_x": np.array(
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 7, 5) input tensor
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0, 13.0, 14.0],
                            [15.0, 16.0, 17.0, 18.0, 19.0],
                            [20.0, 21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0, 29.0],
                            [
                                30.0,
                                31.0,
                                32.0,
                                33.0,
                                34.0,
                            ],
                        ]
                    ]
                ]
            ).astype(np.float32),
            "name": "conv2d_transpose_VALID_padding",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array(
                [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
            ).astype(np.float32),
            "dilation": 1,
            "padding": "VALID",
            "stride": 2,
        },
        {
            "input_x": np.array(
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 7, 5) input tensor
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0, 13.0, 14.0],
                            [15.0, 16.0, 17.0, 18.0, 19.0],
                            [20.0, 21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0, 29.0],
                            [
                                30.0,
                                31.0,
                                32.0,
                                33.0,
                                34.0,
                            ],
                        ]
                    ]
                ]
            ).astype(np.float32),
            "name": "conv2d_transpose_strides_padding",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array(
                [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
            ).astype(np.float32),
            "dilation": 1,
            "padding": 1,
            "stride": 2,
        },
        {
            "input_x": np.array(
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 7, 5) input tensor
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0, 13.0, 14.0],
                            [15.0, 16.0, 17.0, 18.0, 19.0],
                            [20.0, 21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0, 29.0],
                            [
                                30.0,
                                31.0,
                                32.0,
                                33.0,
                                34.0,
                            ],
                        ]
                    ]
                ]
            ).astype(np.float32),
            "name": "conv2d_transpose_strides_no_padding",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array(
                [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
            ).astype(np.float32),
            "dilation": 1,
            "padding": 0,
            "stride": 2,
        },
        {
            "input_x": np.array(
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 7, 5) input tensor
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0, 13.0, 14.0],
                            [15.0, 16.0, 17.0, 18.0, 19.0],
                            [20.0, 21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0, 29.0],
                            [
                                30.0,
                                31.0,
                                32.0,
                                33.0,
                                34.0,
                            ],
                        ]
                    ]
                ]
            ).astype(np.float32),
            "name": "conv2d_transpose_strides_assymetric_padding",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array(
                [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
            ).astype(np.float32),
            "dilation": 1,
            "padding": [1, 1, 0, 1],
            "stride": 2,
        },
        {
            "input_x": np.array(
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 7, 5) input tensor
                            [5.0, 6.0, 7.0, 8.0, 9.0],
                            [10.0, 11.0, 12.0, 13.0, 14.0],
                            [15.0, 16.0, 17.0, 18.0, 19.0],
                            [20.0, 21.0, 22.0, 23.0, 24.0],
                            [25.0, 26.0, 27.0, 28.0, 29.0],
                            [
                                30.0,
                                31.0,
                                32.0,
                                33.0,
                                34.0,
                            ],
                        ]
                    ]
                ]
            ).astype(np.float32),
            "name": "conv2d_transpose_dilation_assymetric_pads_strides",
            "input_shape": [1, 1, 7, 5],
            "kernel": np.array(
                [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
            ).astype(np.float32),
            "dilation": 1,
            "padding": [1, 1, 1, 2],
            "stride": [3, 1],
        },
        {
            "input_x": np.arange(27).astype(np.float32).reshape([1, 3, 3, 3]),
            "name": "depthwise_conv2d_transpose_convolution",
            "input_shape": [1, 3, 3, 3],
            "kernel": np.ones([3, 1, 3, 3]).astype(np.float32),
            "dilation": 1,
            "padding": 1,
            "stride": 1,
            "groups": 3,
            "use_cudnn": False,
        },
    ]
    for test in test_cases:

        paddle_conv2d_transpose(
            test["input_x"],
            test["name"],
            test["input_shape"],
            test["kernel"],
            test["dilation"],
            test["padding"],
            test["stride"],
            1 if "groups" not in test else test["groups"],
            True if "use_cudnn" not in test else test["use_cudnn"],
        )
