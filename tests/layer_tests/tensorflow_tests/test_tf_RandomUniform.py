# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestRandomUniform(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'shape:0' in inputs_info, "Test error: inputs_info must contain `shape`"
        inputs_data = {}
        inputs_data['shape:0'] = np.array(self.shape_value).astype(self.shape_type)
        return inputs_data

    def create_tf_random_uniform_net(self, shape_value, shape_type, dtype, seed, seed2):
        self.shape_value = shape_value
        self.shape_type = shape_type
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            shape = tf.compat.v1.placeholder(shape_type, [len(shape_value)], 'shape')
            tf.raw_ops.RandomUniform(shape=shape, dtype=dtype, seed=seed, seed2=seed2)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('shape_value', [[2], [3, 4], [8, 5, 3]])
    @pytest.mark.parametrize('shape_type', [np.int32, np.int64])
    @pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
    @pytest.mark.parametrize('seed', [312, 50])
    @pytest.mark.parametrize('seed2', [1, 22])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_random_uniform(self, shape_value, shape_type, dtype, seed, seed2,
                            ie_device, precision, ir_version, temp_dir,
                            use_legacy_frontend):
        if dtype == np.float16 or dtype == np.float64:
            pytest.skip('156027: Incorrect specification of RandomUniform for float16 and float64 output type')
        if platform.machine() in ["aarch64", "arm64", "ARM64"]:
            pytest.skip("156055: accuracy error on ARM")
        if ie_device == 'GPU':
            pytest.skip('156056: Accuracy error on GPU')
        self._test(*self.create_tf_random_uniform_net(shape_value, shape_type, dtype, seed, seed2),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
