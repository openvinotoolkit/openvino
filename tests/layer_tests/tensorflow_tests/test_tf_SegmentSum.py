# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSegmentSum(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data:0' in inputs_info, "Test error: inputs_info must contain `data`"
        assert 'segment_ids:0' in inputs_info, "Test error: inputs_info must contain `segment_ids`"
        data_shape = inputs_info['data:0']
        segment_ids_shape = inputs_info['segment_ids:0']
        inputs_data = {}
        inputs_data['data:0'] = np.random.randint(-50, 50, data_shape)
        # segment_ids data must be sorted according to TensorFlow SegmentSum specification
        inputs_data['segment_ids:0'] = np.sort(np.random.randint(0, 20, segment_ids_shape))
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
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_segment_sum_basic(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        if use_legacy_frontend:
            pytest.skip("SegmentSum operation is not supported via legacy frontend.")
        self._test(*self.create_segment_sum_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_different_types = [
        dict(data_shape=[2, 3], segment_ids_shape=[2], data_type=tf.int32, segment_ids_type=tf.int32),
        dict(data_shape=[3, 2], segment_ids_shape=[3], data_type=tf.float64, segment_ids_type=tf.int32),
        dict(data_shape=[3, 1, 2], segment_ids_shape=[3], data_type=tf.float32, segment_ids_type=tf.int64),
        dict(data_shape=[4, 2, 1], segment_ids_shape=[4], data_type=tf.float64, segment_ids_type=tf.int64),
    ]

    @pytest.mark.parametrize("params", test_data_different_types)
    @pytest.mark.nightly
    def test_segment_sum_different_types(self, params, ie_device, precision, ir_version, temp_dir,
                                         use_legacy_frontend):
        if use_legacy_frontend:
            pytest.skip("SegmentSum operation is not supported via legacy frontend.")
        self._test(*self.create_segment_sum_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestSegmentSumComplex(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert "data_real:0" in inputs_info
        assert "data_imag:0" in inputs_info
        assert "segment_ids:0" in inputs_info
        data_real_shape = inputs_info["data_real:0"]
        data_imag_shape = inputs_info["data_imag:0"]
        segment_ids_shape = inputs_info["segment_ids:0"]
        inputs_data = {}
        inputs_data["data_real:0"] = rng.random(data_real_shape).astype(np.float32)
        inputs_data["data_imag:0"] = rng.random(data_imag_shape).astype(np.float32)
        inputs_data["segment_ids:0"] = np.sort(rng.integers(
            low=0, high=data_real_shape[0], size=segment_ids_shape
        ).astype(np.int32))
        return inputs_data

    def create_segment_sum_net(self, data_shape, segment_ids_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            data_real = tf.compat.v1.placeholder(tf.float32,data_shape,'data_real')
            data_imag = tf.compat.v1.placeholder(tf.float32,data_shape,'data_imag')
            data = tf.raw_ops.Complex(real=data_real,imag=data_imag)
            segment_ids = tf.compat.v1.placeholder(tf.int32, segment_ids_shape, "segment_ids")
            segment_sum_output = tf.raw_ops.SegmentSum(data=data, segment_ids=segment_ids)
            tf.raw_ops.Real(input=segment_sum_output)
            tf.raw_ops.Imag(input=segment_sum_output)
            tf.compat.v1.global_variables_initializer().run()
            tf_net = sess.graph_def

        return tf_net, None 

    test_data_basic = [
        dict(data_shape=[2, 3], segment_ids_shape=[2]),
        dict(data_shape=[3, 2], segment_ids_shape=[3]),
        dict(data_shape=[3, 1, 2], segment_ids_shape=[3]),
        dict(data_shape=[4, 2, 1], segment_ids_shape=[4]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_segment_sum(self, params, ie_device, precision, ir_version, temp_dir,
                                         use_legacy_frontend):
        if use_legacy_frontend:
            pytest.skip("SegmentSum operation is not supported via legacy frontend.")
        self._test(*self.create_segment_sum_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
