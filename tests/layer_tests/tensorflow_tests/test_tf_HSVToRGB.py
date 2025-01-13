# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(23345)


class TestHSVToRGB(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'images:0' in inputs_info
        images_shape = inputs_info['images:0']
        inputs_data = {}
        if self.special_case == 'Black Image':
            inputs_data['images:0'] = np.zeros(images_shape).astype(self.input_type)
        elif self.special_case == 'Grayscale Image':
            inputs_data['images:0'] = np.broadcast_to([0, 0, 0.5], images_shape).astype(self.input_type)
        else:
            inputs_data['images:0'] = rng.uniform(0.0, 1.0, images_shape).astype(self.input_type)
        return inputs_data

    def create_hsv_to_rgb_net(self, input_shape, input_type, special_case):
        self.special_case = special_case
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            images = tf.compat.v1.placeholder(input_type, input_shape, 'images')
            tf.raw_ops.HSVToRGB(images=images)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('input_shape', [[3], [5, 3], [4, 5, 3], [5, 21, 21, 3]])
    @pytest.mark.parametrize('input_type', [np.float16, np.float32, np.float64])
    @pytest.mark.parametrize('special_case', [None, 'Black Image', 'Grayscale Image'])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_hsv_to_rgb_basic(self, input_shape, input_type, special_case,
                              ie_device, precision, ir_version, temp_dir,
                              use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip('158898: accuracy issue on GPU')
        self._test(*self.create_hsv_to_rgb_net(input_shape, input_type, special_case),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend, custom_eps=3 * 1e-3)
