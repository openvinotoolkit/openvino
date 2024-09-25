# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(2323534)


class TestAdjustSaturation(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'images:0' in inputs_info
        assert 'scale:0' in inputs_info
        images_shape = inputs_info['images:0']
        inputs_data = {}
        if self.special_case == 'Black Image':
            inputs_data['images:0'] = np.zeros(images_shape).astype(self.input_type)
        elif self.special_case == 'Grayscale Image':
            inputs_data['images:0'] = np.ones(images_shape).astype(self.input_type) * np.random.rand()
        else:
            inputs_data['images:0'] = rng.uniform(0, 1.0, images_shape).astype(self.input_type)
        inputs_data['scale:0'] = rng.uniform(0, 1.0, []).astype(np.float32)
        return inputs_data

    def create_adjust_saturation_net(self, input_shape, input_type, special_case):
        self.special_case = special_case
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            images = tf.compat.v1.placeholder(input_type, input_shape, 'images')
            scale = tf.compat.v1.placeholder(tf.float32, [], 'scale')
            tf.raw_ops.AdjustSaturation(images=images, scale=scale)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('input_shape', [[7, 7, 3], [5, 23, 27, 3], [3, 4, 13, 15, 3]])
    @pytest.mark.parametrize('input_type', [np.float32])
    @pytest.mark.parametrize('special_case', [None, 'Black Image', 'Grayscale Image'])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_adjust_saturation_basic(self, input_shape, input_type, special_case,
                                     ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if ie_device == 'GPU' and input_shape in [[5, 23, 27, 3], [3, 4, 13, 15, 3]]:
            pytest.skip("151264: accuracy error on GPU")
        if platform.machine() in ["aarch64", "arm64", "ARM64"] and input_shape in [[5, 23, 27, 3], [3, 4, 13, 15, 3]]:
            pytest.skip("151263: accuracy error on ARM")
        self._test(*self.create_adjust_saturation_net(input_shape, input_type, special_case),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
