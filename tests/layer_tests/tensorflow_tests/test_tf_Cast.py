# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest


# Testing Cast operation (Initial Implementation)
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/Cast

class TestCastOp(CommonTFLayerTest):
    input_type = np.float32

    # Overload inputs generation to fill dummy input
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(-256, 256, inputs_dict[input]).astype(self.input_type)
        return inputs_dict

    # input_shape - should be an array
    # input_type - type of input value
    # output_type - type of output value
    # truncate - boolean flag of truncation
    # ir_version - common parameter
    # use_legacy_frontend - common parameter
    def create_cast_op_placeholder_const_net(self, input_shape, input_type, output_type, truncate, ir_version,
                                             use_legacy_frontend):
        if (input_type == output_type):
            pytest.skip("Input and output types shouldn't be equal")

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        self.input_type = input_type

        tf_types = {
            np.int32: tf.int32, np.int64: tf.int64,
            np.float16: tf.float16, np.float32: tf.float32, np.float64: tf.float64
        }

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input = tf.compat.v1.placeholder(tf_types[input_type], input_shape, 'Input')

            tf.raw_ops.Cast(x=input_shape, DstT=output_type, Truncate=truncate)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        pytest.param(
            dict(input_shape=[2, 3]),  # Simple test
            marks=pytest.mark.precommit),
        dict(input_shape=[2, 3, 3, 4]),  # Simple test with possible nchw/nhwc
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("input_type", [np.int32, np.int64, np.float16, np.float32, np.float64])
    @pytest.mark.parametrize("output_type", [np.int32, np.int64, np.float16, np.float32, np.float64])
    @pytest.mark.parametrize("truncate", [False, True])
    @pytest.mark.nightly
    def test_cast_op_placeholder_const(self, params, input_type, output_type, truncate, ie_device, precision,
                                       ir_version, temp_dir,
                                       use_legacy_frontend):
        if ie_device == 'GPU' and output_type == np.float64:
            pytest.skip("accuracy mismatch float64 output_type on GPU")
        self._test(*self.create_cast_op_placeholder_const_net(**params, ir_version=ir_version,
                                                              use_legacy_frontend=use_legacy_frontend,
                                                              input_type=input_type,
                                                              output_type=output_type, truncate=truncate),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
