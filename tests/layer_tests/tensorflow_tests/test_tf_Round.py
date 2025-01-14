# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(23536)


class TestRound(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info, "Test error: inputs_info must contain `input`"
        input_shape = inputs_info['input:0']
        inputs_data = {}
        if np.issubdtype(self.input_type, np.floating):
            inputs_data['input:0'] = rng.uniform(-10.0, 10.0, input_shape).astype(self.input_type)
        else:
            inputs_data['input:0'] = rng.integers(-10, 10, input_shape).astype(self.input_type)
        return inputs_data

    def create_tf_round_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf.raw_ops.Round(x=input)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("input_shape", [[6], [2, 5, 3], [10, 5, 1, 5]])
    @pytest.mark.parametrize("input_type", [np.float16, np.float32, np.float64,
                                            np.int8, np.int16, np.int32, np.int64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_round_basic(self, input_shape, input_type, ie_device, precision,
                         ir_version, temp_dir, use_legacy_frontend):
        if input_type in [np.int8, np.int16, np.int32, np.int64]:
            pytest.skip('TensorFlow issue: https://github.com/tensorflow/tensorflow/issues/74789')
        if platform.machine() in ["aarch64", "arm64", "ARM64"] and \
                input_type == np.float32 and input_shape == [10, 5, 1, 5]:
            pytest.skip("150999: Accuracy issue on CPU")
        self._test(*self.create_tf_round_net(input_shape, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
