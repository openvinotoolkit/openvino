# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestDynamicPartition(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data:0' in inputs_info, "Test error: inputs_info must contain `data`"
        assert 'partitions:0' in inputs_info, "Test error: inputs_info must contain `partitions`"
        data_shape = inputs_info['data:0']
        partitions_shape = inputs_info['partitions:0']
        inputs_data = {}
        inputs_data['data:0'] = np.random.randint(-50, 50, data_shape)
        # segment_ids data must be sorted according to TensorFlow SegmentSum specification
        inputs_data['partitions:0'] = np.random.randint(0, 5, partitions_shape)
        return inputs_data

    def create_dynamic_partition_net(self, data_shape, partitions_shape, num_partitions, data_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(data_type, data_shape, 'data')
            partitions = tf.compat.v1.placeholder(tf.int32, partitions_shape, 'partitions')
            dynamic_partition = tf.raw_ops.DynamicPartition(data=data, partitions=partitions,
                                                            num_partitions=num_partitions)
            for ind in range(num_partitions):
                tf.identity(dynamic_partition[ind], name='dynamic_partition_{}'.format(ind))
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(data_shape=[6], partitions_shape=[6], num_partitions=10, data_type=tf.float32),
        dict(data_shape=[4, 3], partitions_shape=[4], num_partitions=8, data_type=tf.float32),
        dict(data_shape=[3, 4, 2], partitions_shape=[3], num_partitions=5, data_type=tf.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_dynamic_partition_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                     use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.xfail('104855')
        if use_legacy_frontend:
            pytest.skip("DynamicPartition operation is not supported via legacy frontend.")
        self._test(*self.create_dynamic_partition_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_other_types = [
        dict(data_shape=[10], partitions_shape=[10], num_partitions=10, data_type=tf.int32),
        dict(data_shape=[7, 3], partitions_shape=[7], num_partitions=8, data_type=tf.int64),
    ]

    @pytest.mark.parametrize("params", test_data_other_types)
    @pytest.mark.nightly
    def test_dynamic_partition_other_types(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.xfail('104855')
        if use_legacy_frontend:
            pytest.skip("DynamicPartition operation is not supported via legacy frontend.")
        self._test(*self.create_dynamic_partition_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
