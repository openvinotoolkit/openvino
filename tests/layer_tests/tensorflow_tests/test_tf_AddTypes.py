# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()


class TestAddTypes(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        if np.issubdtype(self.input_type, np.floating):
            inputs_data['x:0'] = rng.uniform(-5.0, 5.0, x_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            inputs_data['x:0'] = rng.integers(-8, 8, x_shape).astype(self.input_type)
        else:
            inputs_data['x:0'] = rng.integers(0, 8, x_shape).astype(self.input_type)
        return inputs_data

    def create_add_types_net(self, const_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, [], 'x')
            if np.issubdtype(self.input_type, np.floating):
                const_value = rng.uniform(-5.0, 5.0, const_shape).astype(self.input_type)
            elif np.issubdtype(self.input_type, np.signedinteger):
                const_value = rng.integers(-8, 8, const_shape).astype(self.input_type)
            else:
                # test bigger unsigned integer constants and avoid overflow by using smaller input values 0..8
                const_value = rng.integers(128, 240, const_shape).astype(self.input_type)
            const_input = tf.constant(const_value, dtype=input_type)
            tf.raw_ops.Add(x=x, y=const_input)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("const_shape", [[], [2], [3, 4], [3, 2, 1, 4]])
    @pytest.mark.parametrize("input_type", [np.int8, np.uint8, np.int16,
                                            np.int32, np.int64,
                                            np.float16, np.float32, np.float64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_add_types(self, const_shape, input_type, ie_device, precision, ir_version, temp_dir,
                       use_legacy_frontend):
        if ie_device == 'GPU' and input_type == np.int16:
            pytest.skip("accuracy mismatch for int16 on GPU")
        self._test(*self.create_add_types_net(const_shape, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
