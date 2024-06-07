# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest

# Testing Fill operation (Initial Implementation)
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/Fill

class TestFillOps(CommonTFLayerTest):
    stored_shape = []

    # Overload inputs generation to fill dummy input
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.ndarray([len(self.stored_shape)], buffer=np.array(self.stored_shape), dtype=np.int32)
        # Return shape as is
        return inputs_dict

    # input_shape - should be an array
    # value - value which should be set to tensor
    # ir_version - common parameter
    # use_legacy_frontend - common parameter
    def create_fill_ops_placeholder_const_net(self, input_shape, value, ir_version, use_legacy_frontend):
        self.stored_shape = input_shape

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input = tf.compat.v1.placeholder(tf.int32, [len(input_shape)], 'Input')

            tf.raw_ops.Fill(dims = tf_input, value = value)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shape=[2, 3], value=123),
        dict(input_shape=[2, 3, 3, 4], value=123),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_fill_ops_placeholder_const(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_fill_ops_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)