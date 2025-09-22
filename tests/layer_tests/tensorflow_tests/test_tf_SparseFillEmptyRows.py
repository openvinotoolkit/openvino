# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(475912)


class TestSparseFillEmptyRows(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'indices:0' in inputs_info
        assert 'values:0' in inputs_info
        assert 'dense_shape:0' in inputs_info
        assert 'default_value:0' in inputs_info

        values_shape = inputs_info['values:0']

        inputs_data = {}
        if np.issubdtype(self.data_type, np.floating):
            inputs_data['values:0'] = rng.uniform(-5.0, 5.0, values_shape).astype(self.data_type)
            inputs_data['default_value:0'] = np.array(42.0, dtype=self.data_type)
        elif np.issubdtype(self.data_type, np.signedinteger):
            inputs_data['values:0'] = rng.integers(-8, 8, values_shape).astype(self.data_type)
            inputs_data['default_value:0'] = np.array(99, dtype=self.data_type)
        else:
            inputs_data['values:0'] = rng.integers(0, 8, values_shape).astype(self.data_type)
            inputs_data['default_value:0'] = np.array(99, dtype=self.data_type)

        inputs_data['dense_shape:0'] = np.array(self.dense_shape, dtype=np.int64)
        if values_shape[0] == 0:
            inputs_data['indices:0'] = np.zeros((0, 2), dtype=np.int64)
            return inputs_data

        num_rows = self.dense_shape[0]
        num_cols = self.dense_shape[1]
        values_count = values_shape[0]
        
        available_rows = list(range(num_rows))
        if self.empty_rows:
            # Remove specific rows to ensure they're empty
            empty_row_indices = sorted(rng.choice(available_rows, size=self.empty_rows, replace=False))
            for row in empty_row_indices:
                available_rows.remove(row)
        
        row_indices = rng.choice(available_rows, size=values_count, replace=True)
        col_indices = rng.integers(0, num_cols, size=values_count)
        
        indices = np.column_stack((row_indices, col_indices))
        # Sort indices by row, then by column for consistent ordering
        indices = indices[np.lexsort((indices[:, 1], indices[:, 0]))]
        inputs_data['indices:0'] = indices.astype(np.int64)
        
        return inputs_data

    def create_sparse_fill_empty_rows_net(self, data_type, dense_shape, values_count, empty_rows=2):
        self.data_type = data_type
        self.dense_shape = dense_shape
        self.empty_rows = empty_rows
        
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            indices = tf.compat.v1.placeholder(tf.int64, [values_count, 2], 'indices')
            values = tf.compat.v1.placeholder(data_type, [values_count], 'values')
            dense_shape = tf.compat.v1.placeholder(tf.int64, [2], 'dense_shape')
            default_value = tf.compat.v1.placeholder(data_type, [], 'default_value')
            tf.raw_ops.SparseFillEmptyRows(
                indices=indices,
                values=values,
                dense_shape=dense_shape,
                default_value=default_value)
                
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def
        return tf_net, None

    @pytest.mark.parametrize('data_type', [np.float32, np.int32])
    @pytest.mark.parametrize('dense_shape, values_count, empty_rows', [
        [[5, 6], 4, 2],     # Basic case with 2 empty rows
        [[10, 10], 8, 3],   # Larger case with 3 empty rows
        [[3, 4], 0, 3],     # All rows empty
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_sparse_fill_empty_rows(self, data_type,
                                   dense_shape, values_count, empty_rows,
                                   ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip("operation extension is not supported on GPU")
        self._test(*self.create_sparse_fill_empty_rows_net(data_type,
                                                         dense_shape, values_count, empty_rows),
                  ie_device, precision, ir_version, temp_dir=temp_dir)
