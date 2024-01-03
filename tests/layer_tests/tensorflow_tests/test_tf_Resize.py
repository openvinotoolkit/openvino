# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


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
        dict(images_shape=[1, 30, 30, 3], images_type=tf.float32, size_value=[40, 40], align_corners=False,
             half_pixel_centers=False, resize_op=tf.raw_ops.ResizeBilinear),
        dict(images_shape=[1, 30, 30, 3], images_type=tf.float64, size_value=[40, 40], align_corners=False,
             half_pixel_centers=False, resize_op=tf.raw_ops.ResizeBilinear),
        dict(images_shape=[2, 100, 100, 3], images_type=tf.float32, size_value=[40, 40], align_corners=True,
             half_pixel_centers=False, resize_op=tf.raw_ops.ResizeBilinear),
        dict(images_shape=[2, 10, 10, 3], images_type=tf.float32, size_value=[40, 40], align_corners=False,
             half_pixel_centers=True, resize_op=tf.raw_ops.ResizeBilinear),
        dict(images_shape=[2, 40, 40, 3], images_type=tf.uint8, size_value=[10, 10], align_corners=False,
             half_pixel_centers=False, resize_op=tf.raw_ops.ResizeBilinear),
        dict(images_shape=[1, 40, 40, 3], images_type=tf.int32, size_value=[10, 10], align_corners=False,
             half_pixel_centers=True, resize_op=tf.raw_ops.ResizeBilinear),
        # ResizeNearestNeighbor testing
        dict(images_shape=[1, 30, 30, 3], images_type=tf.float32, size_value=[40, 40], align_corners=False,
             half_pixel_centers=False, resize_op=tf.raw_ops.ResizeNearestNeighbor),
        dict(images_shape=[2, 100, 100, 3], images_type=tf.float32, size_value=[40, 40], align_corners=True,
             half_pixel_centers=False, resize_op=tf.raw_ops.ResizeNearestNeighbor),
        dict(images_shape=[2, 10, 10, 3], images_type=tf.float32, size_value=[40, 40], align_corners=False,
             half_pixel_centers=True, resize_op=tf.raw_ops.ResizeNearestNeighbor),
        dict(images_shape=[2, 40, 40, 3], images_type=tf.uint8, size_value=[10, 10], align_corners=False,
             half_pixel_centers=False, resize_op=tf.raw_ops.ResizeNearestNeighbor),
        dict(images_shape=[1, 40, 40, 3], images_type=tf.int32, size_value=[10, 10], align_corners=False,
             half_pixel_centers=True, resize_op=tf.raw_ops.ResizeNearestNeighbor),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_resize_basic(self, params, ie_device, precision, ir_version, temp_dir,
                          use_new_frontend, use_old_api):
        self._test(*self.create_resize_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
