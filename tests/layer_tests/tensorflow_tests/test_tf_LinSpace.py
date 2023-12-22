# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestLinSpace(CommonTFLayerTest):
    def create_lin_space_net(self, num_value):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            start = tf.compat.v1.placeholder(tf.float32, [], 'start')
            stop = tf.compat.v1.placeholder(tf.float32, [], 'stop')
            tf.raw_ops.LinSpace(start=start, stop=stop, num=num_value)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(num_value=2),
        dict(num_value=10),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_lin_space_basic(self, params, ie_device, precision, ir_version, temp_dir,
                             use_new_frontend, use_old_api):
        self._test(*self.create_lin_space_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
