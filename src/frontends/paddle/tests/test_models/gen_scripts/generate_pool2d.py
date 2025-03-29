# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pool2d paddle model generator
#
import numpy as np
import sys
from save_model import saveModel

data_type = "float32"


def pool2d(name: str, x, attrs: dict):
    import paddle

    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=data_type)

        if attrs["pool_type"] == "max":
            out = paddle.nn.functional.max_pool2d(
                node_x,
                kernel_size=attrs["pool_size"],
                stride=attrs["pool_stride"],
                padding=attrs["pool_padding"],
                ceil_mode=attrs["ceil_mode"],
                data_format=attrs["data_format"]
            )
        else:
            out = paddle.nn.functional.avg_pool2d(
                node_x,
                kernel_size=attrs["pool_size"],
                stride=attrs["pool_stride"],
                padding=attrs["pool_padding"],
                ceil_mode=attrs["ceil_mode"],
                data_format=attrs["data_format"]
            )

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feed_vars=[node_x],
            fetchlist=[out],
            inputs=[x],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def adaptive_pool2d(name: str, x, attrs: dict):
    import paddle

    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        if attrs["pool_type"] == "max":
            out = paddle.nn.functional.adaptive_max_pool2d(
                x=node_x,
                output_size=attrs["pool_size"]
            )
        else:
            out = paddle.nn.functional.adaptive_avg_pool2d(
                x=node_x,
                output_size=attrs["pool_size"]
            )

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feed_vars=[node_x],
            fetchlist=[out],
            inputs=[x],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    N, C, H, W = 2, 3, 4, 4
    data = np.arange(N * C * H * W).astype(data_type)
    data_NCHW = data.reshape(N, C, H, W)
    data_NHWC = data.reshape(N, H, W, C)
    # print(data_NCHW, data_NCHW.shape)

    pooling_types = ["max", "avg"]

    # pool2d
    for i, pooling_type in enumerate(pooling_types):
        # example 1:
        # ceil_mode = False
        paddle_attrs = {
            # input=data_NCHW, # shape: [2, 3, 8, 8]
            "pool_size": [3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": [2, 1],  # it is same as pool_padding = [2,2,1,1]
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
        }
        # shape of out_1: [2, 3, 4, 3]
        pool2d(pooling_type + "Pool_test1", data_NCHW, paddle_attrs)

        # Cecilia: there is a bug of PaddlePaddle in this case.
        # example 2:
        # ceil_mode = True (different from example 1)
        paddle_attrs = {
            # input=data_NCHW,
            "pool_size": [3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": [
                [0, 0],
                [0, 0],
                [2, 2],
                [1, 1],
            ],  # it is same as pool_padding = [2,2,1,1]
            "global_pooling": False,
            "ceil_mode": True,
            "exclusive": True,
            "data_format": "NCHW",
        }
        # shape of out_2: [2, 3, 4, 4] which is different from out_1
        pool2d(pooling_type + "Pool_test2", data_NCHW, paddle_attrs)

        # example 3:
        # pool_padding = "SAME" (different from example 1)
        paddle_attrs = {
            # input=data_NCHW,
            "pool_size": [3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": "SAME",
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
        }
        # shape of out_3: [2, 3, 3, 3] which is different from out_1
        pool2d(pooling_type + "Pool_test3", data_NCHW, paddle_attrs)

        # example 4:
        # pool_padding = "VALID" (different from example 1)
        paddle_attrs = {
            # input=data_NCHW,
            "pool_size": [3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": "VALID",
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
        }
        # shape of out_4: [2, 3, 2, 2] which is different from out_1
        pool2d(pooling_type + "Pool_test4", data_NCHW, paddle_attrs)

        # example 5:
        # global_pooling = True (different from example 1)
        # It will be set pool_size = [8,8] and pool_padding = [0,0] actually.
        paddle_attrs = {
            # input=data_NCHW,
            "pool_size": [3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": [2, 1],
            "global_pooling": True,
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
        }
        # shape of out_5: [2, 3, 1, 1] which is different from out_1
        pool2d(pooling_type + "Pool_test5", data_NCHW, paddle_attrs)

        # example 6:
        # data_format = "NHWC" (different from example 1)
        paddle_attrs = {
            # input=data_NHWC, # shape: [2, 8, 8, 3]
            "pool_size": [3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": [2, 1],
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NHWC",
        }
        # shape of out_6: [2, 4, 3, 3] which is different from out_1
        pool2d(pooling_type + "Pool_test6", data_NHWC, paddle_attrs)

        # example 7:
        # pool_size is [9, 9]
        paddle_attrs = {
            # input=data_NCHW,
            "pool_size": [9, 9],
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": [
                [0, 0],
                [0, 0],
                [2, 2],
                [1, 1],
            ],  # it is same as pool_padding = [2,2,1,1]
            "global_pooling": False,
            "ceil_mode": True,
            "exclusive": True,
            "data_format": "NCHW",
        }
        pool2d(pooling_type + "Pool_test7", data_NCHW, paddle_attrs)

        # example 8:
        # pool_padding size is 1
        paddle_attrs = {
            "pool_size": [3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": 2,
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
        }
        pool2d(pooling_type + "Pool_test8", data_NCHW, paddle_attrs)

        # input data for test9 and test10
        N_data1, C_data1, H_data1, W_data1 = 2, 3, 8, 8
        data1 = np.arange(N_data1 * C_data1 * H_data1 * W_data1).astype(data_type)
        data1_NCHW = data1.reshape(N_data1, C_data1, H_data1, W_data1)
        # example 9:
        # pool_padding size is 4: [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]
        paddle_attrs = {
            "pool_size": [3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": [2, 1, 2, 1],
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
        }
        pool2d(pooling_type + "Pool_test9", data1_NCHW, paddle_attrs)

        # example 10:
        # input=data_NCHW and pool_padding is [[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]
        paddle_attrs = {
            "pool_size": [3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": [[0, 0], [0, 0], [2, 1], [2, 1]],
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
        }
        pool2d(pooling_type + "Pool_test10", data1_NCHW, paddle_attrs)

        # example 11:
        # input=data_NCHW and poolsize is the multiply by width & height. pool_padding is [[0,0], [0,0], [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right]]
        paddle_attrs = {
            "pool_size": 9,
            "pool_type": pooling_type,
            "pool_stride": [3, 3],
            "pool_padding": [[0, 0], [0, 0], [2, 1], [2, 1]],
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "data_format": "NCHW",
        }
        pool2d(pooling_type + "Pool_test11", data1_NCHW, paddle_attrs)

    # adaptive_pool2d
    for i, pooling_type in enumerate(pooling_types):
        paddle_attrs = {
            "pool_size": [3, 3],
            "pool_type": pooling_type
        }
        adaptive_pool2d(pooling_type + "AdaptivePool2D_test1", data_NCHW, paddle_attrs)


if __name__ == "__main__":
    main()
