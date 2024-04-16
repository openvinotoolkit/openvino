# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


OPS = {
    "tf.raw_ops.CheckNumerics": tf.raw_ops.CheckNumerics,
    "tf.raw_ops.CheckNumericsV2": tf.raw_ops.CheckNumericsV2
}

class TestCheckNumerics(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        x_shape = inputs_info['x:0']
        assert 'y:0' in inputs_info
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape).astype(self.input_type)
        inputs_data['y:0'] = np.random.randint(-10, 10, y_shape).astype(self.input_type)
        return inputs_data

    def create_check_numerics_net(self, input_shape, input_type, op):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            y = tf.compat.v1.placeholder(input_type, input_shape, 'y')
            z = tf.raw_ops.AddV2(x=x, y=y)
            op(tensor=z, message="z is invalid")
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        [[2, 6], np.float32, 'tf.raw_ops.CheckNumerics'],
        [[3, 4, 5], np.float32, 'tf.raw_ops.CheckNumericsV2'],
    ]

    @pytest.mark.parametrize("input_shape, input_type, op", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_check_numerics_basic(self, input_shape, input_type, op, ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        params = dict(input_shape=input_shape, input_type=input_type, op=OPS[op])
        self._test(*self.create_check_numerics_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
