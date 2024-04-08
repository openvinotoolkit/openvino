# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestUnpack(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape).astype(self.input_type)
        return inputs_data

    def create_unpack_net(self, input_shape, num, axis, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            type_map = {
                np.float32: tf.float32,
                np.int32: tf.int32,
            }
            assert input_type in type_map, "Test error: need to update type_map"
            tf_type = type_map[input_type]
            x = tf.compat.v1.placeholder(tf_type, input_shape, 'x')
            if axis is not None:
                unpack = tf.raw_ops.Unpack(value=x, num=num, axis=axis)
            else:
                unpack = tf.raw_ops.Unpack(value=x, num=num)
            for ind in range(num):
                tf.identity(unpack[ind], name="output_" + str(ind))
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 3, 4], num=3, axis=1, input_type=np.float32),
        dict(input_shape=[3, 4], num=3, axis=None, input_type=np.int32),
        dict(input_shape=[4, 2, 3], num=2, axis=-2, input_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_unpack_basic(self, params, ie_device, precision, ir_version, temp_dir,
                          use_legacy_frontend):
        self._test(*self.create_unpack_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
