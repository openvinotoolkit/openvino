# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


OPS = {
    'tf.raw_ops.ResizeBilinear': tf.raw_ops.ResizeBilinear,
    'tf.raw_ops.ResizeNearestNeighbor': tf.raw_ops.ResizeNearestNeighbor,
}

class TestResize(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'images' in inputs_info, "Test error: inputs_info must contain `x`"
        images_shape = inputs_info['images']
        inputs_data = {}
        inputs_data['images'] = np.random.randint(0, 10, images_shape)
        return inputs_data

    def create_resize_net(self, images_shape, images_type, size_value, align_corners, half_pixel_centers,
                          resize_op):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            images = tf.compat.v1.placeholder(images_type, images_shape, 'images')
            size = tf.constant(size_value, dtype=tf.int32)
            resize_op(images=images, size=size, align_corners=align_corners,
                      half_pixel_centers=half_pixel_centers)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        # ResizeBilinear testing
        [[1, 30, 30, 3], tf.float32, [40, 40], False, False, 'tf.raw_ops.ResizeBilinear'],
        [[1, 30, 30, 3], tf.float64, [40, 40], False, False, 'tf.raw_ops.ResizeBilinear'],
        [[2, 100, 100, 3], tf.float32, [40, 40], True, False, 'tf.raw_ops.ResizeBilinear'],
        [[2, 10, 10, 3], tf.float32, [40, 40], False, True, 'tf.raw_ops.ResizeBilinear'],
        [[2, 40, 40, 3], tf.uint8, [10, 10], False, False, 'tf.raw_ops.ResizeBilinear'],
        [[1, 40, 40, 3], tf.int32, [10, 10], False, True, 'tf.raw_ops.ResizeBilinear'],
        # ResizeNearestNeighbor testing
        [[1, 30, 30, 3], tf.float32, [40, 40], False, False, 'tf.raw_ops.ResizeNearestNeighbor'],
        [[2, 100, 100, 3], tf.float32, [40, 40], True, False, 'tf.raw_ops.ResizeNearestNeighbor'],
        [[2, 10, 10, 3], tf.float32, [40, 40], False, True, 'tf.raw_ops.ResizeNearestNeighbor'],
        [[2, 40, 40, 3], tf.uint8, [10, 10], False, False, 'tf.raw_ops.ResizeNearestNeighbor'],
        [[1, 40, 40, 3], tf.int32, [10, 10], False,True, 'tf.raw_ops.ResizeNearestNeighbor'],
    ]

    @pytest.mark.parametrize("images_shape, images_type, size_value, align_corners, half_pixel_centers, resize_op", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_resize_basic(self, images_shape, images_type, size_value, align_corners, half_pixel_centers, resize_op, ie_device, precision, ir_version, temp_dir, use_new_frontend):
        params = dict(images_shape=images_shape, images_type=images_type, size_value=size_value, align_corners=align_corners, half_pixel_centers=half_pixel_centers, resize_op=OPS[resize_op])
        self._test(*self.create_resize_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
