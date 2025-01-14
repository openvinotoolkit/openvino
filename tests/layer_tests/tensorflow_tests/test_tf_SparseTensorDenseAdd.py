# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(475912)


class TestSparseTensorDenseAdd(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'a_indices:0' in inputs_info
        assert 'a_values:0' in inputs_info
        assert 'b:0' in inputs_info

        a_values_shape = inputs_info['a_values:0']
        b_shape = inputs_info['b:0']

        inputs_data = {}
        if np.issubdtype(self.data_type, np.floating):
            inputs_data['a_values:0'] = rng.uniform(-5.0, 5.0, a_values_shape).astype(self.data_type)
            inputs_data['b:0'] = rng.uniform(-5.0, 5.0, b_shape).astype(self.data_type)
        elif np.issubdtype(self.data_type, np.signedinteger):
            inputs_data['a_values:0'] = rng.integers(-8, 8, a_values_shape).astype(self.data_type)
            inputs_data['b:0'] = rng.integers(-8, 8, b_shape).astype(self.data_type)
        else:
            inputs_data['a_values:0'] = rng.integers(0, 8, a_values_shape).astype(self.data_type)
            inputs_data['b:0'] = rng.integers(0, 8, b_shape).astype(self.data_type)

        a_rows_num = self.a_shape[0]
        a_cols_num = self.a_shape[1]

        # generate all possible indices
        all_indices = []
        for row_ind in range(0, a_rows_num):
            for col_ind in range(0, a_cols_num):
                all_indices.append([row_ind, col_ind])
        inputs_data['a_indices:0'] = rng.choice(all_indices, self.nnz, replace=False).astype(self.indices_type)

        return inputs_data

    def create_sparse_tensor_dense_add_net(self, data_type, indices_type,
                                               a_shape, b_shape, nnz):
        self.data_type = data_type
        self.indices_type = indices_type
        self.a_shape = a_shape
        self.nnz = nnz
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            a_indices = tf.compat.v1.placeholder(indices_type, [nnz, 2], 'a_indices')
            a_values = tf.compat.v1.placeholder(data_type, [nnz], 'a_values')
            a_shape = tf.constant(a_shape, dtype=indices_type)
            b = tf.compat.v1.placeholder(data_type, b_shape, 'b')
            tf.raw_ops.SparseTensorDenseAdd(
                a_indices=a_indices,
                a_values=a_values,
                a_shape=a_shape,
                b=b)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('data_type', [np.float32, np.float64, np.int32])
    @pytest.mark.parametrize('indices_type', [np.int32, np.int64])
    @pytest.mark.parametrize('a_shape, b_shape, nnz', [
        [[4, 10], [4, 10], 8],
        [[5, 5], [5, 5], 3],
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_sparse_tensor_dense_add(self, data_type, indices_type,
                                    a_shape, b_shape, nnz,
                                    ie_device, precision, ir_version, temp_dir,
                                    use_legacy_frontend):
        self._test(*self.create_sparse_tensor_dense_add_net(data_type, indices_type,
                                                            a_shape, b_shape, nnz),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
