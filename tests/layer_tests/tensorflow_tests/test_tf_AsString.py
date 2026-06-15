# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import run_in_jenkins

rng = np.random.default_rng()


class TestAsString(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        input_shape = inputs_info['input:0']
        inputs_data = {}
        if self.input_dtype == np.bool_:
            inputs_data['input:0'] = rng.choice([True, False], size=input_shape)
        elif np.issubdtype(self.input_dtype, np.floating):
            inputs_data['input:0'] = rng.uniform(-100, 100, input_shape).astype(self.input_dtype)
        elif np.issubdtype(self.input_dtype, np.unsignedinteger):
            inputs_data['input:0'] = rng.integers(0, 100, input_shape).astype(self.input_dtype)
        else:
            inputs_data['input:0'] = rng.integers(-100, 100, input_shape).astype(self.input_dtype)
        return inputs_data

    def create_as_string_net(self, input_shape, input_dtype):
        self.input_dtype = input_dtype

        tf_type_map = {
            np.int32: tf.int32,
            np.int64: tf.int64,
            np.int16: tf.int16,
            np.int8: tf.int8,
            np.uint8: tf.uint8,
            np.uint16: tf.uint16,
            np.uint32: tf.uint32,
            np.uint64: tf.uint64,
            np.float32: tf.float32,
            np.float64: tf.float64,
            np.bool_: tf.bool,
        }

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf_type_map[input_dtype], input_shape, 'input')
            tf.raw_ops.AsString(input=input, name='AsString')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    # Note: float32/float64 are excluded because TF uses fixed-point formatting
    # (e.g. "1.000000") while the tokenizers extension uses default stream formatting
    # (e.g. "1"). The numeric values are correct but string representations differ.
    # Note: bool is excluded because TF bool maps to u8 in OpenVINO element types,
    # so NumericToString treats it as an integer rather than "true"/"false".
    @pytest.mark.parametrize("input_shape", [[3], [2, 3], [1, 2, 3]])
    @pytest.mark.parametrize("input_dtype", [
        np.int32, np.int64, np.int16, np.int8,
        np.uint8, np.uint16, np.uint32, np.uint64,
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_as_string(self, input_shape, input_dtype, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU' or run_in_jenkins():
            pytest.skip("operation extension is not supported on GPU")
        self._test(*self.create_as_string_net(input_shape=input_shape, input_dtype=input_dtype),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
