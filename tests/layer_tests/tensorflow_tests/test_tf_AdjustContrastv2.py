# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(2323534)


class TestAdjustContrastv2(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'images:0' in inputs_info
        assert 'contrast_factor:0' in inputs_info
        inputs_data = {}
        images_shape = inputs_info['images:0']
        inputs_data['images:0'] = rng.uniform(0, 1.0, images_shape).astype(self.input_type)
        inputs_data['contrast_factor:0'] = rng.uniform(0, 1.0, []).astype(np.float32)
        return inputs_data

    def create_adjust_contrast_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            images = tf.compat.v1.placeholder(input_type, input_shape, 'images')
            contrast_factor = tf.compat.v1.placeholder(tf.float32, [], 'contrast_factor')
            tf.raw_ops.AdjustContrastv2(images=images, contrast_factor=contrast_factor)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('input_shape', [[10, 20, 3], [5, 25, 15, 2], [3, 4, 8, 10, 4]])
    @pytest.mark.parametrize('input_type', [np.float32])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_adjust_contrast_basic(self, input_shape, input_type, ie_device, precision, ir_version, temp_dir,
                                   use_legacy_frontend):
        self._test(*self.create_adjust_contrast_net(input_shape, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
