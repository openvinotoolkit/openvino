# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import mix_array_with_value, run_in_jenkins

rng = np.random.default_rng()


class TestLookupTableFindOps(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'keys:0' in inputs_info, "Test error: inputs_info must contain `x`"
        keys_shape = inputs_info['keys:0']
        inputs_data = {}
        if np.issubdtype(self.keys_type, np.integer):
            data = rng.choice(self.all_keys, keys_shape)
            inputs_data['keys:0'] = mix_array_with_value(data, self.invalid_key)
        elif self.keys_type == str:
            data = rng.choice(self.all_keys + [self.invalid_key], keys_shape)
            inputs_data['keys:0'] = data
        else:
            raise "Unsupported type {}".format(self.keys_type)

        return inputs_data

    def create_lookup_table_find_net(self, hash_table_type, keys_shape, keys_type, values_type,
                                     all_keys, all_values, default_value, invalid_key):
        hash_table_op = tf.raw_ops.HashTable if hash_table_type == 0 else tf.raw_ops.HashTableV2
        import_table_op = tf.raw_ops.LookupTableImport if hash_table_type == 0 else tf.raw_ops.LookupTableImportV2
        lookup_table_op = tf.raw_ops.LookupTableFind if hash_table_type == 0 else tf.raw_ops.LookupTableFindV2

        self.keys_type = keys_type
        self.all_keys = all_keys
        self.invalid_key = invalid_key
        if keys_type == str:
            keys_type = tf.string
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            keys = tf.compat.v1.placeholder(keys_type, keys_shape, 'keys')
            all_keys = tf.constant(all_keys, dtype=keys_type)
            all_values = tf.constant(all_values, dtype=values_type)
            default_value = tf.constant(default_value, dtype=values_type)
            hash_table = hash_table_op(key_dtype=keys_type, value_dtype=values_type)
            import_hash_table = import_table_op(table_handle=hash_table, keys=all_keys,
                                                values=all_values)
            with tf.control_dependencies([import_hash_table]):
                lookup_table_op(table_handle=hash_table, keys=keys,
                                default_value=default_value,
                                name='LookupTableFind')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data = [
        dict(keys_type=np.int32, values_type=np.float32, all_keys=[0, 1, 2, 3, 4, 5],
             all_values=[2.0, 13.0, -2.0, 0.0, 3.0, 1.0], default_value=-100.0, invalid_key=-100),
        dict(keys_type=np.int64, values_type=np.int32, all_keys=[0, 1, 2, 3, 4, 5],
             all_values=[2, 13, -2, 0, 3, 1], default_value=-100, invalid_key=-100),
        dict(keys_type=np.int32, values_type=np.float32, all_keys=[2, 0, 3, -2, 4, 10],
             all_values=[2.0, 13.0, -2.0, 0.0, 3.0, 1.0], default_value=-100.0, invalid_key=1000),
        dict(keys_type=np.int64, values_type=np.float32, all_keys=[2, 0, 3, -2, 4, 10],
             all_values=[2.0, 13.0, -2.0, 0.0, 3.0, 1.0], default_value=-100.0, invalid_key=1000),
        dict(keys_type=np.int32, values_type=tf.string, all_keys=[20, 10, 33, -22, 44, 11],
             all_values=['PyTorch', 'TensorFlow', 'JAX', 'Lightning', 'MindSpore', 'OpenVINO'],
             default_value='UNKNOWN', invalid_key=1000),
        dict(keys_type=str, values_type=np.int64,
             all_keys=['PyTorch', 'TensorFlow', 'JAX', 'Lightning', 'MindSpore', 'OpenVINO'],
             all_values=[200, 100, 0, -3, 10, 1],
             default_value=0, invalid_key='AbraCadabra'),
        dict(keys_type=str, values_type=np.int32,
             all_keys=['First sentence', 'Second one', '', 'Third', 'Fourth Sentence', 'etc.'],
             all_values=[-1, 2, 0, -3, 0, 1],
             default_value=100, invalid_key='AbraCadabra'),
    ]

    @pytest.mark.parametrize("hash_table_type", [0, 1])
    @pytest.mark.parametrize("keys_shape", [[], [2], [3, 4], [3, 2, 1, 4]])
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_lookup_table_find(self, hash_table_type, keys_shape, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        if ie_device == 'GPU' or run_in_jenkins():
            pytest.skip("operation extesion is not supported on GPU or "
                        "No layout format available for gather:LookupTableFind issue")
        self._test(*self.create_lookup_table_find_net(hash_table_type=hash_table_type,
                                                      keys_shape=keys_shape, **params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
