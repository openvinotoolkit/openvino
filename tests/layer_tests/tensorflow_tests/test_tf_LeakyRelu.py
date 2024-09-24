# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(54654675)


class TestLeakyRelu(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'features:0' in inputs_info, "Test error: inputs_info must contain `features`"
        features_shape = inputs_info['features:0']
        inputs_data = {}
        inputs_data['features:0'] = rng.uniform(-5.0, 5.0, features_shape).astype(self.features_type)
        return inputs_data

    def create_leaky_relu_net(self, features_shape, features_type, alpha):
        self.features_type = features_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            features = tf.compat.v1.placeholder(features_type, features_shape, 'features')
            tf.raw_ops.LeakyRelu(features=features, alpha=alpha)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('features_shape', [[], [2], [3, 1], [2, 4, 5]])
    @pytest.mark.parametrize('features_type', [np.float16, np.float32, np.float64])
    @pytest.mark.parametrize('alpha', [None, -2.0, 0.0, 1.0, 2.0])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_leaky_relu(self, features_shape, features_type, alpha,
                        ie_device, precision, ir_version,
                        temp_dir, use_legacy_frontend):
        self._test(*self.create_leaky_relu_net(features_shape, features_type, alpha),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
