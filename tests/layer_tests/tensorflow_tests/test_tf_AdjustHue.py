# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(2323534)


class TestAdjustHue(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'images:0' in inputs_info
        if self.special_case == 'Black Image':
            images_shape = inputs_info['images:0']
            inputs_data = {}
            inputs_data['images:0'] = np.zeros(images_shape).astype(self.input_type)
        elif self.special_case == 'Grayscale Image':
            images_shape = inputs_info['images:0']
            inputs_data = {}
            inputs_data['images:0'] = np.ones(images_shape).astype(self.input_type) * np.random.rand()
        else:
            images_shape = inputs_info['images:0']
            inputs_data = {}
            inputs_data['images:0'] = rng.uniform(0, 1.0, images_shape).astype(self.input_type)
        inputs_data['delta:0'] = rng.uniform(0, 1.0, []).astype(np.float32)

        return inputs_data

    def create_adjust_hue_net(self, input_shape, input_type, special_case):
        self.special_case = special_case
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            images = tf.compat.v1.placeholder(input_type, input_shape, 'images')
            delta = tf.compat.v1.placeholder(np.float32, [], 'delta')
            tf.raw_ops.AdjustHue(images=images, delta=delta)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('input_shape', [[2, 5, 3], [2, 4, 4, 3], [3, 4, 12, 12, 3]])
    @pytest.mark.parametrize('input_type', [np.float32])
    @pytest.mark.parametrize('special_case', [None, 'Black Image', 'Grayscale Image'])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_adjust_hue_basic(self, input_shape, input_type, special_case,
                              ie_device, precision, ir_version, temp_dir,
                              use_legacy_frontend):
        self._test(*self.create_adjust_hue_net(input_shape, input_type, special_case),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
