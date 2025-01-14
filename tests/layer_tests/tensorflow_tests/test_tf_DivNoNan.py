# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import mix_array_with_value

rng = np.random.default_rng(23235)


class TestDivNoNan(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        assert 'y:0' in inputs_info
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['x:0'] = rng.uniform(-5.0, 5.0, x_shape).astype(self.input_type)
        # provide zeros in y input
        y_data = rng.uniform(-5.0, 5.0, y_shape).astype(self.input_type)
        y_data = mix_array_with_value(y_data, 0.0)
        inputs_data['y:0'] = y_data
        return inputs_data

    def create_div_no_nan_net(self, x_shape, y_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, x_shape, 'x')
            y = tf.compat.v1.placeholder(input_type, y_shape, 'y')
            tf.raw_ops.DivNoNan(x=x, y=y)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('x_shape', [[], [4], [3, 4], [2, 3, 4]])
    @pytest.mark.parametrize('y_shape', [[2, 3, 4]])
    @pytest.mark.parametrize('input_type', [np.float16, np.float32, np.float64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_div_no_nan_basic(self, x_shape, y_shape, input_type,
                              ie_device, precision, ir_version,
                              temp_dir, use_legacy_frontend):
        self._test(*self.create_div_no_nan_net(x_shape, y_shape, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
