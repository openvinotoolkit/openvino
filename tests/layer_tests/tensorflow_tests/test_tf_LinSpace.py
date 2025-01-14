# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestLinSpace(CommonTFLayerTest):
    def _prepare_input(self, _):
        inputs_data = {}
        inputs_data['start:0'] = np.array(self.start_value, dtype=self.input_type)
        inputs_data['stop:0'] = np.array(self.stop_value, dtype=self.input_type)
        inputs_data['num:0'] = np.array(self.num_value, dtype=self.num_type)
        return inputs_data

    def create_lin_space_net(self, start_value, stop_value, num_value, input_type, num_type):
        self.start_value = start_value
        self.stop_value = stop_value
        self.num_value = num_value
        self.input_type = input_type
        self.num_type = num_type

        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            start = tf.compat.v1.placeholder(input_type, [], 'start')
            stop = tf.compat.v1.placeholder(input_type, [], 'stop')
            num = tf.compat.v1.placeholder(num_type, [], 'num')
            tf.raw_ops.LinSpace(start=start, stop=stop, num=num)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('start_value', [-10.0, -5.0, 0.0, 5.0, 10.0])
    @pytest.mark.parametrize('stop_value', [-10.0, -5.0, 0.0, 5.0, 10.0])
    @pytest.mark.parametrize('num_value', [1, 5, 100])
    @pytest.mark.parametrize('input_type', [np.float32, np.float64])
    @pytest.mark.parametrize('num_type', [np.int32, np.int64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_lin_space_basic(self, start_value, stop_value, num_value, input_type, num_type,
                             ie_device, precision, ir_version, temp_dir,
                             use_legacy_frontend):
        self._test(*self.create_lin_space_net(start_value, stop_value, num_value, input_type, num_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
