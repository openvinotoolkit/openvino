# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()


class TestPlaceholder(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        tensor_name = list(inputs_info.keys())[0]
        x_shape = inputs_info[tensor_name]
        inputs_data = {}
        if self.input_type == str or self.input_type == np.str_:
            strings_dictionary = ['first', 'second sentence', ' sentence 3 three', '34ferf466 23435* ', '汉语句子',
                                  'Oferta polska',
                                  'предложение по-русски']
            inputs_data[tensor_name] = rng.choice(strings_dictionary, x_shape)
        elif np.issubdtype(self.input_type, np.floating):
            inputs_data[tensor_name] = rng.uniform(-5.0, 5.0, x_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            inputs_data[tensor_name] = rng.integers(-8, 8, x_shape).astype(self.input_type)
        else:
            inputs_data[tensor_name] = rng.integers(0, 8, x_shape).astype(self.input_type)
        return inputs_data

    def create_placeholder_net(self, input_shape, input_type):
        dtype = input_type
        if input_type == str or input_type == np.str_:
            dtype = tf.string
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.raw_ops.Placeholder(dtype=dtype, shape=input_shape, name='x')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("input_shape", [[], [2], [3, 4], [3, 2, 1, 4]])
    @pytest.mark.parametrize("input_type", [np.int8, np.uint8, np.int16,
                                            np.int32, np.int64,
                                            np.float16, np.float32, np.float64, str, np.str_])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_placeholder(self, input_shape, input_type, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        if ie_device == 'GPU' and (input_type == str or input_type == np.str_):
            pytest.skip("string tensor is not supported on GPU")
        self._test(*self.create_placeholder_net(input_shape, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
