# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

from common.utils.tf_utils import mix_two_arrays_with_several_values

rng = np.random.default_rng(235345)


class TestXlog1py(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        assert 'y:0' in inputs_info
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['x:0'] = rng.uniform(-5.0, 5.0, x_shape).astype(self.input_type)
        inputs_data['y:0'] = rng.uniform(0.5, 5.0, y_shape).astype(self.input_type)
        # to mix input data so it has inf and nan values for log(y) at positions multiplied by zero x
        inputs_data['x:0'], inputs_data['y:0'] = \
            mix_two_arrays_with_several_values(inputs_data['x:0'], inputs_data['y:0'],
                                               [[0.0, -1.0], [0.0, -10.0], [0.0, np.inf], [0.0, np.nan]], rng)
        return inputs_data

    def create_xlog1py_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            y = tf.compat.v1.placeholder(input_type, input_shape, 'y')
            tf.raw_ops.Xlog1py(x=x, y=y)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('input_shape', [[20, 5], [2, 3, 4]])
    @pytest.mark.parametrize('input_type', [np.float16, np.float32, np.float64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_xlog1py_basic(self, input_shape, input_type, ie_device, precision, ir_version, temp_dir,
                           use_legacy_frontend):
        self._test(*self.create_xlog1py_net(input_shape, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
