# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestToBool(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape).astype(np.float32)
        
        return inputs_data

    def create_tobool_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            tf.raw_ops.ToBool(input=x)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[10, 20], input_type=np.float32),
        dict(input_shape=[2, 3, 4], input_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_to_bool_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                         use_legacy_frontend):
        self._test(*self.create_tobool_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)