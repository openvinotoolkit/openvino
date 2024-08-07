# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

class TestHSVToRGB(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'images:0' in inputs_info
        if self.special_case == "Black Image":
            images_shape = inputs_info['images:0']
            inputs_data = {}
            inputs_data['images:0'] = np.zeros(images_shape).astype(self.input_type) 
        elif self.special_case == "Grayscale Image":
            images_shape = inputs_info['images:0']
            inputs_data = {}
            inputs_data['images:0'] = np.broadcast_to([0, 0, 0.5], images_shape).astype(self.input_type)
        else:
            images_shape = inputs_info['images:0']
            inputs_data = {} 
            inputs_data['images:0'] = np.random.rand(*images_shape).astype(self.input_type)
            
        return inputs_data

    def create_hsv_to_rgb_net(self, input_shape, input_type, special_case=False):
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

    # Each input is a tensor of with values in [0,1].
    # The last dimension must be size 3.
    test_data_basic = [
        dict(input_shape=[7, 7, 3], input_type=np.float32, special_case="Black Image"),
        dict(input_shape=[7, 7, 3], input_type=np.float32, special_case="Grayscale Image"),
        dict(input_shape=[5, 5, 3], input_type=np.float32),
        dict(input_shape=[5, 23, 27, 3], input_type=np.float64),
        dict(input_shape=[3, 4, 13, 15, 3], input_type=np.float64),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ['arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'],
                       reason='Ticket - 126314, 132699')
    def test_hsv_to_rgb_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                   use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("Accuracy mismatch on GPU")
        self._test(*self.create_hsv_to_rgb_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
