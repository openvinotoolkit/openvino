# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import run_in_jenkins

rng = np.random.default_rng()


class TestRaggedTensorToSparse(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'rt_dense_values:0' in inputs_info, "Test error: inputs_info must contain `values`"
        rt_dense_values_shape = inputs_info['rt_dense_values:0']
        inputs_data = {}
        if np.issubdtype(self.rt_dense_values_type, np.floating):
            inputs_data['rt_dense_values:0'] = rng.uniform(-5.0, 5.0, rt_dense_values_shape).astype(
                self.rt_dense_values_type)
        elif np.issubdtype(self.rt_dense_values_type, np.signedinteger):
            inputs_data['rt_dense_values:0'] = rng.integers(-8, 8, rt_dense_values_shape).astype(
                self.rt_dense_values_type)
        else:
            inputs_data['rt_dense_values:0'] = rng.integers(0, 8, rt_dense_values_shape).astype(
                self.rt_dense_values_type)
        return inputs_data

    def create_ragged_tensor_to_sparse_net(self, rt_dense_values_shape, rt_dense_values_type,
                                           rt_nested_splits):
        self.rt_dense_values_type = rt_dense_values_type
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            rt_dense_values = tf.compat.v1.placeholder(rt_dense_values_type, rt_dense_values_shape, 'rt_dense_values')
            sparse_tensor = tf.raw_ops.RaggedTensorToSparse(rt_nested_splits=rt_nested_splits,
                                                            rt_dense_values=rt_dense_values)
            tf.identity(sparse_tensor[0], name='sparse_indices')
            # needed to avoid skip connection
            tf.math.add(sparse_tensor[1], 2)
            tf.identity(sparse_tensor[2], name='sparse_dense_shape')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('rt_dense_values_shape', [[21]])
    @pytest.mark.parametrize('rt_dense_values_type', [np.float32, np.int32, np.int64])
    @pytest.mark.parametrize('rt_nested_splits', [[[0, 0, 1, 6, 8, 8, 15, 20, 21, 21]],
                                                  [[0, 21]],
                                                  [[0, 1, 2, 3, 4, 5, 21]]
                                                  ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_ragged_tensor_to_sparse(self, rt_dense_values_shape, rt_dense_values_type, rt_nested_splits,
                                     ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if ie_device == 'GPU' or run_in_jenkins():
            pytest.skip("operation extension is not supported on GPU")
        self._test(*self.create_ragged_tensor_to_sparse_net(rt_dense_values_shape=rt_dense_values_shape,
                                                            rt_dense_values_type=rt_dense_values_type,
                                                            rt_nested_splits=rt_nested_splits),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
