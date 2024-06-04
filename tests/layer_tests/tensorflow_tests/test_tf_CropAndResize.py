# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestCropAndResize(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'image:0' in inputs_info, "Test error: inputs_info must contain `image`"
        inputs_data = {}
        image_shape = inputs_info['image:0']
        inputs_data['image:0'] = np.random.randint(-10, 10, image_shape).astype(np.float32)

        if 'boxes:0' in inputs_info:
            boxes_shape = inputs_info['boxes:0']
            inputs_data['boxes:0'] = np.random.randint(0, 1.2, boxes_shape).astype(np.float32)
        if 'box_ind:0' in inputs_info:
            box_ind_shape = inputs_info['box_ind:0']
            inputs_data['box_ind:0'] = np.random.randint(0, image_shape[0], box_ind_shape).astype(np.int32)

        return inputs_data

    def create_crop_and_resize_net(self, image_shape, num_boxes, crop_size_value, method, extrapolation_value):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            image = tf.compat.v1.placeholder(tf.float32, image_shape, 'image')
            boxes = tf.compat.v1.placeholder(tf.float32, [num_boxes, 4], 'boxes')
            box_ind = tf.compat.v1.placeholder(tf.int32, [num_boxes], 'box_ind')
            crop_size = tf.constant(crop_size_value, dtype=tf.int32)
            tf.raw_ops.CropAndResize(image=image, boxes=boxes, box_ind=box_ind, crop_size=crop_size, method=method,
                                     extrapolation_value=extrapolation_value)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(image_shape=[5, 45, 30, 3], num_boxes=10, crop_size_value=[8, 5], method="bilinear",
             extrapolation_value=0.0),
        pytest.param(dict(image_shape=[1, 80, 40, 3], num_boxes=10, crop_size_value=[20, 15], method="bilinear",
                          extrapolation_value=1.0),
                     marks=pytest.mark.xfail(reason="102603")),
        pytest.param(dict(image_shape=[20, 10, 50, 2], num_boxes=5, crop_size_value=[4, 50], method="nearest",
                          extrapolation_value=0.0),
                     marks=pytest.mark.xfail(reason="102603")),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(platform.machine() in ["aarch64", "arm64", "ARM64"],
                       reason='Ticket - 122716')
    def test_crop_and_resize_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                   use_legacy_frontend):
        self._test(*self.create_crop_and_resize_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
