# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import run_in_jenkins

rng = np.random.default_rng()


class TestRaggedTensorToTensor(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'values:0' in inputs_info, "Test error: inputs_info must contain `values`"
        values_shape = inputs_info['values:0']
        inputs_data = {}
        if np.issubdtype(self.values_type, np.floating):
            inputs_data['values:0'] = rng.uniform(-5.0, 5.0, values_shape).astype(self.values_type)
        elif np.issubdtype(self.values_type, np.signedinteger):
            inputs_data['values:0'] = rng.integers(-8, 8, values_shape).astype(self.values_type)
        else:
            inputs_data['values:0'] = rng.integers(0, 8, values_shape).astype(self.values_type)
        return inputs_data

    def create_ragged_tensor_to_tensor_net(self, shape_type, shape_value, values_shape, values_type, default_value,
                                           row_partition_tensors, row_partition_types):
        self.values_type = values_type
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            values = tf.compat.v1.placeholder(values_type, values_shape, 'values')
            shape = tf.constant(shape_value, dtype=shape_type)
            default_value = tf.constant(default_value, dtype=values_type)
            tf.raw_ops.RaggedTensorToTensor(shape=shape, values=values, default_value=default_value,
                                            row_partition_tensors=row_partition_tensors,
                                            row_partition_types=row_partition_types)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('shape_type', [np.int32, np.int64])
    @pytest.mark.parametrize('shape_value', [[4, 8], [-1, 64], [5, -1], [-1, -1]])
    @pytest.mark.parametrize('values_shape', [[40], [100]])
    @pytest.mark.parametrize('values_type', [np.float32, np.int32, np.int64])
    @pytest.mark.parametrize('default_value', [-1, 0])
    @pytest.mark.parametrize('row_partition_tensors', [[[0, 1, 6, 8, 15, 20, 21]]])
    @pytest.mark.parametrize('row_partition_types', [["ROW_SPLITS"]])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_ragged_tensor_to_tensor(self, shape_type, shape_value, values_shape, values_type, default_value,
                                     row_partition_tensors, row_partition_types,
                                     ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if ie_device == 'GPU' or run_in_jenkins():
            pytest.skip("operation extension is not supported on GPU")
        self._test(*self.create_ragged_tensor_to_tensor_net(shape_type=shape_type, shape_value=shape_value,
                                                            values_shape=values_shape, values_type=values_type,
                                                            default_value=default_value,
                                                            row_partition_tensors=row_partition_tensors,
                                                            row_partition_types=row_partition_types),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestRaggedTensorToTensorRowIds(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'values:0' in inputs_info, "Test error: inputs_info must contain `values`"
        values_shape = inputs_info['values:0']
        inputs_data = {}
        if np.issubdtype(self.values_type, np.floating):
            inputs_data['values:0'] = rng.uniform(-5.0, 5.0, values_shape).astype(self.values_type)
        elif np.issubdtype(self.values_type, np.signedinteger):
            inputs_data['values:0'] = rng.integers(-8, 8, values_shape).astype(self.values_type)
        else:
            inputs_data['values:0'] = rng.integers(0, 8, values_shape).astype(self.values_type)
        return inputs_data

    def create_ragged_tensor_to_tensor_net(self, shape_type, shape_value, values_shape, values_type, default_value,
                                           row_partition_tensors, row_partition_types):
        self.values_type = values_type
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            values = tf.compat.v1.placeholder(values_type, values_shape, 'values')
            shape = tf.constant(shape_value, dtype=shape_type)
            default_value = tf.constant(default_value, dtype=values_type)
            tf.raw_ops.RaggedTensorToTensor(shape=shape, values=values, default_value=default_value,
                                            row_partition_tensors=row_partition_tensors,
                                            row_partition_types=row_partition_types)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('shape_type', [np.int32, np.int64])
    @pytest.mark.parametrize('shape_value', [[4, 8], [-1, 64], [5, -1], [-1, -1]])
    @pytest.mark.parametrize('values_shape', [[10]])
    @pytest.mark.parametrize('values_type', [np.float32, np.int32, np.int64])
    @pytest.mark.parametrize('default_value', [-1, 0])
    @pytest.mark.parametrize('row_partition_tensors', [[20, [1, 2, 3, 3, 4, 8, 8, 9, 9, 9]]])
    @pytest.mark.parametrize('row_partition_types', [["FIRST_DIM_SIZE", "VALUE_ROWIDS"]])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_ragged_tensor_to_tensor(self, shape_type, shape_value, values_shape, values_type, default_value,
                                     row_partition_tensors, row_partition_types,
                                     ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if ie_device == 'GPU' or run_in_jenkins():
            pytest.skip("operation extension is not supported on GPU")
        self._test(*self.create_ragged_tensor_to_tensor_net(shape_type=shape_type, shape_value=shape_value,
                                                            values_shape=values_shape, values_type=values_type,
                                                            default_value=default_value,
                                                            row_partition_tensors=row_partition_tensors,
                                                            row_partition_types=row_partition_types),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
