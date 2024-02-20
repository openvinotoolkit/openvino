# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestUnsortedSegmentSum(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data' in inputs_info, "Test error: inputs_info must contain `data`"
        assert 'segment_ids' in inputs_info, "Test error: inputs_info must contain `segment_ids`"
        data_shape = inputs_info['data']
        segment_ids_shape = inputs_info['segment_ids']
        inputs_data = {}
        inputs_data['data'] = np.random.randint(-50, 50, data_shape).astype(self.data_type)
        # segment_ids can have negative values
        inputs_data['segment_ids'] = np.random.randint(-self.num_segments_val, self.num_segments_val, segment_ids_shape)
        return inputs_data

    def create_unsorted_segment_sum_net(self, data_shape, segment_ids_shape, num_segments_val, data_type,
                                        segment_ids_type, num_segments_type):
        self.data_type = data_type
        self.segment_ids_type = segment_ids_type
        self.num_segments_val = num_segments_val
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(data_type, data_shape, 'data')
            segment_ids = tf.compat.v1.placeholder(segment_ids_type, segment_ids_shape, 'segment_ids')
            num_segments = tf.constant(num_segments_val, dtype=num_segments_type, shape=[])
            tf.raw_ops.UnsortedSegmentSum(data=data, segment_ids=segment_ids, num_segments=num_segments)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(data_shape=[8], segment_ids_shape=[8], num_segments_val=10),
        dict(data_shape=[10, 4], segment_ids_shape=[10], num_segments_val=5),
        dict(data_shape=[5, 6, 7], segment_ids_shape=[5], num_segments_val=100),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.parametrize("data_type", [
        np.float32, np.int32
    ])
    @pytest.mark.parametrize("segment_ids_type", [
        np.int32, np.int64
    ])
    @pytest.mark.parametrize("num_segments_type", [
        np.int32, np.int64
    ])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_unsorted_segment_sum_basic(self, params, data_type, segment_ids_type, num_segments_type, ie_device,
                                        precision, ir_version, temp_dir,
                                        use_new_frontend):
        if not use_new_frontend:
            pytest.skip("UnsortedSegmentSum operation is not supported via legacy frontend.")
        self._test(
            *self.create_unsorted_segment_sum_net(**params, data_type=data_type, segment_ids_type=segment_ids_type,
                                                  num_segments_type=num_segments_type),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend)
