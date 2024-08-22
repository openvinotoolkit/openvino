# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(872173)


class TestTensorScatterUpdate(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'tensor:0' in inputs_info
        assert 'indices:0' in inputs_info
        assert 'updates:0' in inputs_info

        tensor_shape = inputs_info['tensor:0']
        updates_shape = inputs_info['updates:0']
        indices_shape = inputs_info['indices:0']

        inputs_data = {}
        if np.issubdtype(self.data_type, np.floating):
            inputs_data['tensor:0'] = rng.uniform(-5.0, 5.0, tensor_shape).astype(self.data_type)
            inputs_data['updates:0'] = rng.uniform(-5.0, 5.0, updates_shape).astype(self.data_type)
        elif np.issubdtype(self.data_type, np.signedinteger):
            inputs_data['tensor:0'] = rng.integers(-8, 8, tensor_shape).astype(self.data_type)
            inputs_data['updates:0'] = rng.integers(-8, 8, updates_shape).astype(self.data_type)
        else:
            inputs_data['tensor:0'] = rng.integers(0, 8, tensor_shape).astype(self.data_type)
            inputs_data['updates:0'] = rng.integers(0, 8, updates_shape).astype(self.data_type)
        
        tensor_rows = tensor_shape[0]
        tensor_cols = tensor_shape[1]
        indices_cols = indices_shape[1]

        all_indices = []
        if indices_cols == 1:
            for row_ind in range(0, tensor_rows):
                all_indices.append([row_ind])
        else:
            for row_ind in range(0, tensor_rows):
                for col_ind in range(0, tensor_cols):
                    all_indices.append([row_ind, col_ind])

        inputs_data['indices:0'] = rng.choice(all_indices, indices_shape[0], replace=False).astype(self.indices_type)

        return inputs_data

    def create_tensor_scatter_update_net(self, data_type, indices_type,
                                        tensor_shape, updates_shape, indices_shape):
        self.data_type = data_type
        self.indices_type = indices_type
        self.tensor_shape = tensor_shape
        self.updates_shape = updates_shape
        self.indices_shape = indices_shape
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            indices = tf.compat.v1.placeholder(indices_type, indices_shape, 'indices')
            tensor = tf.compat.v1.placeholder(data_type, tensor_shape, 'tensor')
            updates = tf.compat.v1.placeholder(data_type, updates_shape, 'updates')
            tf.raw_ops.TensorScatterUpdate(
                tensor=tensor,
                indices=indices,
                updates=updates)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('data_type', [np.float32, np.float64, np.int32])
    @pytest.mark.parametrize('indices_type', [np.int32, np.int64])
    @pytest.mark.parametrize('tensor_shape, updates_shape, indices_shape', [
        [[10, 5], [2], [2, 2]],
        [[4, 4, 4], [2, 4, 4], [2, 1]],
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_sparse_tensor_dense_add(self, data_type, indices_type,
                                     tensor_shape, updates_shape, indices_shape,
                                     ie_device, precision, ir_version, temp_dir,
                                     use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("149830: ScatterNDUpdate-15 is not supported on GPU")
        self._test(*self.create_tensor_scatter_update_net(data_type, indices_type,
                                                          tensor_shape, updates_shape, indices_shape),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
