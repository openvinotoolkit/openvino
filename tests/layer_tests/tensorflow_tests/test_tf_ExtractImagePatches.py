# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestExtractImagePatches(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        # generate elements so that the input tensor may contain repeating elements
        assert 'images:0' in inputs_info, "Test error: inputs_info must contain `images`"
        images_shape = inputs_info['images:0']
        inputs_data = {}
        inputs_data['images:0'] = np.random.randint(-10, 10, images_shape).astype(np.float32)
        return inputs_data

    def create_extract_image_patches_net(self, images_shape, ksizes, strides, rates, padding):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            images = tf.compat.v1.placeholder(tf.float32, images_shape, 'images')
            tf.raw_ops.ExtractImagePatches(images=images, ksizes=ksizes, strides=strides, rates=rates, padding=padding)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_basic = [
        # TensorFlow supports patching only across spatial dimensions
        dict(images_shape=[2, 110, 50, 4], ksizes=[1, 20, 30, 1], strides=[1, 5, 5, 1], rates=[1, 1, 1, 1]),
        dict(images_shape=[3, 30, 40, 3], ksizes=[1, 5, 10, 1], strides=[1, 3, 1, 1], rates=[1, 4, 3, 1]),
    ]

    @pytest.mark.parametrize("params", test_basic)
    @pytest.mark.parametrize("padding", ["SAME", "VALID"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_extract_image_patches_basic(self, params, padding, ie_device, precision, ir_version, temp_dir,
                                         use_legacy_frontend):
        if ie_device == 'GPU' and padding == 'SAME':
            pytest.skip("accuracy mismatch for VALID padding on GPU")
        self._test(*self.create_extract_image_patches_net(**params, padding=padding),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
