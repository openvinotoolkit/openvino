# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pool3d paddle model generator
#
import numpy as np
import sys
from save_model import saveModel

data_type = "float32"


def pool3d(name: str, x, attrs: dict):
    import paddle

    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        if attrs["pool_type"] == "max":
            out = paddle.nn.functional.max_pool3d(
                node_x,
                kernel_size=attrs["pool_size"],
                stride=attrs["pool_stride"],
                padding=attrs["pool_padding"],
                ceil_mode=attrs["ceil_mode"],
                data_format=attrs["data_format"],
                return_mask=attrs["return_mask"],
            )
        else:
            out = paddle.nn.functional.avg_pool3d(
                node_x,
                kernel_size=attrs["pool_size"],
                stride=attrs["pool_stride"],
                padding=attrs["pool_padding"],
                ceil_mode=attrs["ceil_mode"],
                data_format=attrs["data_format"],
            )

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        if attrs["return_mask"]:
            outs = exe.run(feed={"x": x}, fetch_list=[out[0], out[1]])
            saveModel(
                name,
                exe,
                feed_vars=[node_x],
                fetchlist=[out[0], out[1]],
                inputs=[x],
                outputs=[outs[0], outs[1]],
                target_dir=sys.argv[1],
            )
        else:
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


def adaptive_pool3d(name: str, x, attrs: dict):
    import paddle

    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        if attrs["pool_type"] == "max":
            out = paddle.nn.functional.adaptive_max_pool3d(
                x=node_x,
                output_size=attrs["pool_size"],
                return_mask=attrs["return_mask"],
            )
        else:
            out = paddle.nn.functional.adaptive_avg_pool3d(
                x=node_x, output_size=attrs["pool_size"]
            )

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        if attrs["return_mask"]:
            outs = exe.run(feed={"x": x}, fetch_list=[out[0], out[1]])
            saveModel(
                name,
                exe,
                feed_vars=[node_x],
                fetchlist=[out[0], out[1]],
                inputs=[x],
                outputs=[outs[0], outs[1]],
                target_dir=sys.argv[1],
            )
        else:
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
    N, C, D, H, W = 2, 3, 4, 4, 4
    data = np.arange(N * C * D * H * W).astype(data_type)
    data_NCDHW = data.reshape(N, C, D, H, W)
    data_NDHWC = data.reshape(N, D, H, W, C)

    pooling_types = ["max", "avg"]

    for i, pooling_type in enumerate(pooling_types):
        # example 1:
        # ceil_mode = False
        paddle_attrs = {
            # input=data_NCDHW, # shape: [2, 3, 4, 4, 4]
            "pool_size": [3, 3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3, 3],
            "pool_padding": [
                1,
                2,
                1,
            ],  # it is same as pool_padding = [1, 1, 2, 2, 1, 1]
            "ceil_mode": False,
            "exclusive": True,
            "return_mask": False,
            "data_format": "NCDHW",
        }
        pool3d(pooling_type + "3d" + "Pool_test1", data_NCDHW, paddle_attrs)

        # example 2:
        # ceil_mode = True (different from example 1)
        paddle_attrs = {
            # input=data_NCDHW,
            "pool_size": [3, 3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3, 3],
            "pool_padding": [
                [0, 0],
                [0, 0],
                [1, 1],
                [2, 2],
                [1, 1],
            ],  # it is same as pool_padding = [1, 1, 2, 2, 1, 1]
            "ceil_mode": True,
            "exclusive": True,
            "return_mask": False,
            "data_format": "NCDHW",
        }
        pool3d(pooling_type + "3d" + "Pool_test2", data_NCDHW, paddle_attrs)

        # example 3:
        # pool_padding = "SAME" (different from example 1)
        paddle_attrs = {
            # input=data_NCDHW,
            "pool_size": [3, 3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3, 3],
            "pool_padding": "SAME",
            "ceil_mode": False,
            "exclusive": True,
            "return_mask": False,
            "data_format": "NCDHW",
        }
        pool3d(pooling_type + "3d" + "Pool_test3", data_NCDHW, paddle_attrs)

        # example 4:
        # pool_padding = "VALID" (different from example 1)
        paddle_attrs = {
            # input=data_NCDHW,
            "pool_size": [3, 3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3, 3],
            "pool_padding": "VALID",
            "ceil_mode": False,
            "exclusive": True,
            "return_mask": False,
            "data_format": "NCDHW",
        }
        pool3d(pooling_type + "3d" + "Pool_test4", data_NCDHW, paddle_attrs)

        # example 5:
        # data_format = "NDHWC" (different from example 1)
        paddle_attrs = {
            # input=data_NDHWC, # shape: [2, 4, 4, 4, 3]
            "pool_size": [3, 3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3, 3],
            "pool_padding": [1, 2, 1],
            "ceil_mode": False,
            "exclusive": True,
            "return_mask": False,
            "data_format": "NDHWC",
        }
        # NOT support data_format = "NDHWC" now
        pool3d(pooling_type + "3d" + "Pool_test5", data_NDHWC, paddle_attrs)

        # example 6:
        # pool_padding size is 1
        paddle_attrs = {
            "pool_size": [3, 3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3, 3],
            "pool_padding": 2,
            "ceil_mode": False,
            "exclusive": True,
            "return_mask": False,
            "data_format": "NCDHW",
        }
        pool3d(pooling_type + "3d" + "Pool_test6", data_NCDHW, paddle_attrs)

        # input data for test7 and test8
        N_data1, C_data1, D_data1, H_data1, W_data1 = 2, 3, 8, 8, 8
        data1 = np.arange(N_data1 * C_data1 * D_data1 * H_data1 * W_data1).astype(
            data_type
        )
        data1_NCDHW = data1.reshape(N_data1, C_data1, D_data1, H_data1, W_data1)
        data1_NDHWC = data1.reshape(N_data1, D_data1, H_data1, W_data1, C_data1)
        # example 7:
        # pool_padding size is 6: [pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]
        paddle_attrs = {
            "pool_size": [3, 3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3, 3],
            "pool_padding": [1, 2, 1, 1, 2, 1],
            "ceil_mode": False,
            "exclusive": True,
            "return_mask": False,
            "data_format": "NCDHW",
        }
        pool3d(pooling_type + "3d" + "Pool_test7", data1_NCDHW, paddle_attrs)

        # example 8:
        paddle_attrs = {
            "pool_size": [3, 3, 3],
            "pool_type": pooling_type,
            "pool_stride": [3, 3, 3],
            "pool_padding": [[0, 0], [0, 0], [1, 2], [2, 1], [2, 1]],
            "ceil_mode": False,
            "exclusive": True,
            "return_mask": False,
            "data_format": "NCDHW",
        }
        pool3d(pooling_type + "3d" + "Pool_test8", data1_NCDHW, paddle_attrs)

        # example 9:
        paddle_attrs = {
            "pool_size": 9,
            "pool_type": pooling_type,
            "pool_stride": [3, 3, 3],
            "pool_padding": [[0, 0], [0, 0], [2, 1], [1, 2], [1, 2]],
            "ceil_mode": False,
            "exclusive": True,
            "return_mask": False,
            "data_format": "NCDHW",
        }
        pool3d(pooling_type + "3d" + "Pool_test9", data1_NCDHW, paddle_attrs)

        # example 10:
        paddle_attrs = {
            "pool_size": 9,
            "pool_type": pooling_type,
            "pool_stride": 3,
            "pool_padding": [[0, 0], [2, 2], [1, 2], [2, 2], [0, 0]],
            "ceil_mode": False,
            "return_mask": False,
            "data_format": "NDHWC",
        }
        pool3d(pooling_type + "3d" + "Pool_test10", data1_NDHWC, paddle_attrs)

    paddle_attrs = {
        "pool_size": 9,
        "pool_type": "max",
        "pool_stride": 3,
        "pool_padding": [3, 3, 3],
        "ceil_mode": False,
        "return_mask": True,
        "data_format": "NCDHW",
    }
    pool3d("max3dRetureMask", data_NCDHW, paddle_attrs)

    # adaptive_pool3
    for i, pooling_type in enumerate(pooling_types):
        paddle_attrs = {
            "pool_size": [2, 2, 2],
            "pool_type": pooling_type,
            "return_mask": False,
        }
        adaptive_pool3d(pooling_type + "AdaptivePool3D_test1", data_NCDHW, paddle_attrs)
        paddle_attrs = {"pool_size": 2, "pool_type": pooling_type, "return_mask": False}
        adaptive_pool3d(pooling_type + "AdaptivePool3D_test2", data_NCDHW, paddle_attrs)
        paddle_attrs = {
            "pool_size": 1,  # global pooling case
            "pool_type": pooling_type,
            "return_mask": False,
        }
        adaptive_pool3d(pooling_type + "AdaptivePool3D_test3", data_NCDHW, paddle_attrs)
        paddle_attrs = {
            "pool_size": 1,  # global pooling case
            "pool_type": pooling_type,
            "return_mask": True,
        }
        adaptive_pool3d(pooling_type + "AdaptivePool3D_test4", data_NCDHW, paddle_attrs)


if __name__ == "__main__":
    main()
