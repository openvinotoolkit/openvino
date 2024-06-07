# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# roi_align paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import ops
import sys


def make_rois(batch_size, width, height, pooled_width, pooled_height, spatial_scale, roi_per_batch):
    rois = []
    rois_num = []
    for bno in range(batch_size):
        for i in range(roi_per_batch):
            x1 = np.random.randint(
                0, width // spatial_scale - pooled_width)
            y1 = np.random.randint(
                0, height // spatial_scale - pooled_height)

            x2 = np.random.randint(x1 + pooled_width,
                                   width // spatial_scale)
            y2 = np.random.randint(
                y1 + pooled_height, height // spatial_scale)

            roi = [x1, y1, x2, y2]
            rois.append(roi)
        rois_num.append(len(rois))
    rois = np.array(rois).astype("float32")
    rois_num = np.array(rois_num).astype("int32")

    return rois, rois_num


def roi_align(name: str, x_data, rois_data, rois_num_data, pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x = paddle.static.data(
            name='x', shape=x_data.shape, dtype=x_data.dtype)
        rois = paddle.static.data(
            name='rois', shape=rois_data.shape, dtype=rois_data.dtype)
        rois_num = paddle.static.data(
            name='rois_num', shape=rois_num_data.shape, dtype=rois_num_data.dtype)
        if paddle.__version__ >= "2.6.0":
            out = paddle.vision.ops.roi_align(x=x,
                                              boxes=rois,
                                              boxes_num=rois_num,
                                              output_size=(pooled_height, pooled_width),
                                              spatial_scale=spatial_scale,
                                              sampling_ratio=sampling_ratio,
                                              aligned=aligned)
        else:
            out = ops.roi_align(input=x,
                                rois=rois,
                                output_size=(pooled_height, pooled_width),
                                spatial_scale=spatial_scale,
                                sampling_ratio=sampling_ratio,
                                rois_num=rois_num,
                                aligned=aligned)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x_data, 'rois': rois_data, 'rois_num': rois_num_data},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[x, rois, rois_num], fetchlist=[out],inputs=[
                  x_data, rois_data, rois_num_data], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    batch_size = 1
    channels = 3
    height = 8
    width = 6

    x_dim = (batch_size, channels, height, width)
    x = np.random.random(x_dim).astype('float32')

    spatial_scale = 1.0 / 2.0
    pooled_height = 2
    pooled_width = 2
    sampling_ratio = -1
    aligned = False

    roi_per_batch = 1
    rois, rois_num = make_rois(batch_size, width, height, pooled_width,
                               pooled_height, spatial_scale, roi_per_batch)

    roi_align("roi_align_test", x, rois, rois_num, pooled_height,
              pooled_width, spatial_scale, sampling_ratio, aligned)

    batch_size = 1
    channels = 3
    height = 8
    width = 6

    x_dim = (batch_size, channels, height, width)
    x = np.random.random(x_dim).astype('float32')

    spatial_scale = 1.0 / 2.0
    pooled_height = 2
    pooled_width = 2
    sampling_ratio = 2
    aligned = True

    roi_per_batch = 2
    rois, rois_num = make_rois(batch_size, width, height, pooled_width,
                               pooled_height, spatial_scale, roi_per_batch)

    roi_align("roi_align_test2", x, rois, rois_num, pooled_height,
              pooled_width, spatial_scale, sampling_ratio, aligned)


if __name__ == "__main__":
    main()
