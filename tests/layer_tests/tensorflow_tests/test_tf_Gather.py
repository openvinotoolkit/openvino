# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()


class TestGather(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'params:0' in inputs_info
        assert 'indices:0' in inputs_info
        params_shape = inputs_info['params:0']
        indices_shape = inputs_info['indices:0']
        inputs_data = {}
        if self.params_type == str or self.params_type == np.str_:
            strings_dictionary = ['first', 'second sentence', ' sentence 3 three', '34ferf466 23435* ']
            inputs_data['params:0'] = rng.choice(strings_dictionary, params_shape)
        else:
            inputs_data['params:0'] = rng.integers(-50, 50, params_shape).astype(self.params_type)
        inputs_data['indices:0'] = rng.integers(0, self.max_index, indices_shape).astype(self.indices_type)
        return inputs_data

    def create_gather_net(self, params_shape, params_type, indices_shape, indices_type, axis_value, batch_dims,
                          operation_type):
        self.params_type = params_type
        if params_type == str or params_type == np.str_:
            params_type = tf.string
        self.indices_type = indices_type
        if batch_dims is None:
            batch_dims = 0
        if axis_value is None:
            axis_value = 0
        axis_norm = axis_value
        if axis_norm < 0:
            axis_norm += len(params_shape)
        assert 0 <= axis_norm < len(params_shape), "Incorrect `axis` value for the test case"
        self.max_index = params_shape[axis_norm]

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            params = tf.compat.v1.placeholder(params_type, params_shape, 'params')
            indices = tf.compat.v1.placeholder(indices_type, indices_shape, 'indices')
            if operation_type == "Gather":
                tf.raw_ops.Gather(params=params, indices=indices)
            elif operation_type == "GatherV2":
                axis = tf.constant(axis_value, dtype=tf.int32)
                tf.raw_ops.GatherV2(params=params, indices=indices, axis=axis, batch_dims=batch_dims)
            else:
                assert False, "Incorrect operation type is tested"

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_precommit = [
        dict(params_shape=[4, 6], indices_shape=[], axis_value=None, batch_dims=None, operation_type='Gather'),
        dict(params_shape=[3, 4, 6], indices_shape=[3, 4], axis_value=None, batch_dims=None, operation_type='Gather'),
        dict(params_shape=[5, 4, 3], indices_shape=[5, 2, 1], axis_value=2, batch_dims=1, operation_type='GatherV2'),
        dict(params_shape=[3, 2, 6, 4], indices_shape=[3, 2, 1, 3], axis_value=-1, batch_dims=-2,
             operation_type='GatherV2'),
    ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.parametrize("params_type", [np.float32, np.int32, str, np.str_])
    @pytest.mark.parametrize("indices_type", [np.int32, np.int64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_gather(self, params, params_type, indices_type, ie_device, precision, ir_version, temp_dir,
                    use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("timeout issue on GPU")
        self._test(*self.create_gather_net(**params, params_type=params_type, indices_type=indices_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
