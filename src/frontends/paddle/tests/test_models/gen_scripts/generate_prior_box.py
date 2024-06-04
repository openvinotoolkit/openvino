# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# prior_box paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def prior_box(name: str, input_data, image_data, attrs: dict):
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        Input = paddle.static.data(
            name='Input', shape=input_data.shape, dtype=input_data.dtype)
        Image = paddle.static.data(
            name='Image', shape=image_data.shape, dtype=image_data.dtype)

        if paddle.__version__ >= '2.0.0':
            box, var = paddle.vision.ops.prior_box(Input,
                                                   Image,
                                                   min_sizes=attrs['min_sizes'],
                                                   max_sizes=attrs['max_sizes'],
                                                   aspect_ratios=attrs['aspect_ratios'],
                                                   variance=attrs['variance'],
                                                   flip=attrs['flip'],
                                                   clip=attrs['clip'],
                                                   steps=attrs['steps'],
                                                   offset=attrs['offset'],
                                                   name=None,
                                                   min_max_aspect_ratios_order=attrs['min_max_aspect_ratios_order'])
        else:
            box, var = paddle.fluid.layers.prior_box(Input,
                                                     Image,
                                                     min_sizes=attrs['min_sizes'],
                                                     max_sizes=attrs['max_sizes'],
                                                     aspect_ratios=attrs['aspect_ratios'],
                                                     variance=attrs['variance'],
                                                     flip=attrs['flip'],
                                                     clip=attrs['clip'],
                                                     steps=attrs['steps'],
                                                     offset=attrs['offset'],
                                                     name=None,
                                                     min_max_aspect_ratios_order=attrs['min_max_aspect_ratios_order'])

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'Input': input_data, 'Image': image_data},
            fetch_list=[box, var])

        # Save inputs in order of OpenVINO model, to facilite Fuzzy test,
        # which accepts inputs and outputs in this order as well.
        saveModel(name, exe, feed_vars=[Input, Image], fetchlist=[box, var],
                  inputs=[input_data, image_data], outputs=outs, target_dir=sys.argv[1])
    return outs


if __name__ == "__main__":

    prior_box_attrs_default = {
        'name': "prior_box_default",
        'min_sizes': np.array([2, 4]).astype('float32').tolist(),
        'max_sizes': np.array([5, 10]).astype('float32').tolist(),
        'aspect_ratios': [2.0, 3.0],
        'flip': True,
        'clip': True,
        'steps': np.array([1.25, 1.25]).astype('float32').tolist(),
        'offset': 0.5,
        'variance': np.array([0.1, 0.1, 0.2, 0.2], dtype=float).flatten(),
        'min_max_aspect_ratios_order': False
    }

    prior_box_max_sizes_none = {
        'name': "prior_box_max_sizes_none",
        'min_sizes': np.array([2, 4]).astype('float32').tolist(),
        'max_sizes': None,
        'aspect_ratios': [2.0, 3.0],
        'flip': True,
        'clip': True,
        'steps': np.array([1.25, 1.25]).astype('float32').tolist(),
        'offset': 0.5,
        'variance': np.array([0.1, 0.1, 0.2, 0.2], dtype=float).flatten(),
        'min_max_aspect_ratios_order': False
    }

    prior_box_flip_clip_false = {
        'name': "prior_box_flip_clip_false",
        'min_sizes': np.array([2, 4]).astype('float32').tolist(),
        'max_sizes': np.array([5, 10]).astype('float32').tolist(),
        'aspect_ratios': [2.0, 3.0],
        'flip': False,
        'clip': False,
        'steps': np.array([1.25, 1.25]).astype('float32').tolist(),
        'offset': 0.5,
        'variance': np.array([0.1, 0.1, 0.2, 0.2], dtype=float).flatten(),
        'min_max_aspect_ratios_order': False
    }

    prior_box_attrs_mmar_order_true = {
        'name': "prior_box_attrs_mmar_order_true",
        'min_sizes': np.array([2, 4]).astype('float32').tolist(),
        'max_sizes': np.array([5, 10]).astype('float32').tolist(),
        'aspect_ratios': [2.0, 3.0],
        'flip': True,
        'clip': True,
        'steps': np.array([1.25, 1.25]).astype('float32').tolist(),
        'offset': 0.5,
        'variance': np.array([0.1, 0.1, 0.2, 0.2], dtype=float).flatten(),
        'min_max_aspect_ratios_order': True
    }

    prior_box_attrs_list = [prior_box_attrs_default,
                            prior_box_max_sizes_none, prior_box_flip_clip_false, prior_box_attrs_mmar_order_true]

    layer_w = 32
    layer_h = 32

    image_w = 40
    image_h = 40

    input_channels = 2
    image_channels = 3
    batch_size = 10

    input_data = np.random.random(
        (batch_size, input_channels, layer_w,
         layer_h)).astype('float32')

    image_data = np.random.random(
        (batch_size, image_channels, image_w,
         image_h)).astype('float32')

    for item in prior_box_attrs_list:
        pred_paddle = prior_box(item['name'], input_data, image_data, item)
