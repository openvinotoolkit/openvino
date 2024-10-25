# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestRandomUniformInt(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'shape:0' in inputs_info, "Test error: inputs_info must contain `shape`"
        inputs_data = {}
        inputs_data['shape:0'] = np.array(self.shape_value).astype(self.shape_type)
        return inputs_data

    def create_tf_random_uniform_int_net(self, shape_value, shape_type, minval_type,
                                         minval_value, maxval_value, seed, seed2):
        self.shape_value = shape_value
        self.shape_type = shape_type
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            shape = tf.compat.v1.placeholder(shape_type, [len(shape_value)], 'shape')
            minval = tf.constant(minval_value, dtype=minval_type)
            maxval = tf.constant(maxval_value, dtype=minval_type)
            tf.raw_ops.RandomUniformInt(shape=shape, minval=minval, maxval=maxval,
                                        seed=seed, seed2=seed2)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('shape_value', [[2], [3, 4], [8, 5, 3]])
    @pytest.mark.parametrize('shape_type', [np.int32, np.int64])
    @pytest.mark.parametrize('minval_type', [np.int32, np.int64])
    @pytest.mark.parametrize('minval_value,maxval_value', [(0, 10), (-5, 4), (-10, -1)])
    @pytest.mark.parametrize('seed', [312, 50])
    @pytest.mark.parametrize('seed2', [1, 22])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_random_uniform_int(self, shape_value, shape_type, minval_type, minval_value, maxval_value, seed, seed2,
                                ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if minval_type == np.int64:
            pytest.skip('156027: Incorrect specification of RandomUniform for np.int64 output type')
        self._test(*self.create_tf_random_uniform_int_net(shape_value, shape_type, minval_type,
                                                          minval_value, maxval_value, seed, seed2),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
