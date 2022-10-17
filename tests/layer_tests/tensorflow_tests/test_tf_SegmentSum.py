# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSegmentSum(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data' in inputs_info, "Test error: inputs_info must contain `data`"
        assert 'segment_ids' in inputs_info, "Test error: inputs_info must contain `segment_ids`"
        data_shape = inputs_info['data']
        segment_ids_shape = inputs_info['segment_ids']
        inputs_data = {}
        inputs_data['data'] = np.random.randint(-50, 50, data_shape)
        # segment_ids data must be sorted according to TensorFlow SegmentSum specification
        inputs_data['segment_ids'] = np.sort(np.random.randint(0, 20, segment_ids_shape))
        return inputs_data

    def create_segment_sum_net(self, data_shape, segment_ids_shape, data_type, segment_ids_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(data_type, data_shape, 'data')
            segment_ids = tf.compat.v1.placeholder(segment_ids_type, segment_ids_shape, 'segment_ids')
            tf.math.segment_sum(data, segment_ids)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(data_shape=[8], segment_ids_shape=[8], data_type=tf.float32, segment_ids_type=tf.int32),
        dict(data_shape=[3, 7], segment_ids_shape=[3], data_type=tf.float32, segment_ids_type=tf.int32),
        dict(data_shape=[4, 3, 2], segment_ids_shape=[4], data_type=tf.float32, segment_ids_type=tf.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    def test_segment_sum_basic(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(*self.create_segment_sum_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_different_types = [
        dict(data_shape=[2, 3], segment_ids_shape=[2], data_type=tf.int32, segment_ids_type=tf.int32),
        dict(data_shape=[3, 2], segment_ids_shape=[3], data_type=tf.float64, segment_ids_type=tf.int32),
        dict(data_shape=[3, 1, 2], segment_ids_shape=[3], data_type=tf.float32, segment_ids_type=tf.int64),
        dict(data_shape=[4, 2, 1], segment_ids_shape=[4], data_type=tf.float64, segment_ids_type=tf.int64),
    ]

    @pytest.mark.parametrize("params", test_data_different_types)
    @pytest.mark.nightly
    def test_segment_sum_different_types(self, params, ie_device, precision, ir_version, temp_dir,
                                         use_new_frontend, use_old_api):
        self._test(*self.create_segment_sum_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
