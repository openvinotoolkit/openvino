# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sys import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestTensorListLength(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape).astype(self.input_type)
        return inputs_data

    def create_tensor_list_length(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            tensor_list = tf.raw_ops.TensorListFromTensor(tensor=x,
                                                          element_shape=tf.constant(input_shape[1:], dtype=tf.int32))
            tf.raw_ops.TensorListLength(input_handle=tensor_list)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[7], input_type=np.float32),
        dict(input_shape=[10, 20], input_type=np.float32),
        dict(input_shape=[2, 3, 4], input_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_tensor_list_length_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_tensor_list_length(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestTensorListLengthEmptyList(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        inputs_data = {}
        inputs_data['tensor_list_size:0'] = np.array([self.tensor_list_size], dtype=np.int32)
        return inputs_data

    def create_tensor_list_length_empty_list(self, tensor_list_size, element_shape):
        self.tensor_list_size = tensor_list_size
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tensor_list_size = tf.compat.v1.placeholder(tf.int32, [1], 'tensor_list_size')
            tf_element_shape = tf.constant(element_shape, dtype=tf.int32)
            tensor_shape = tf.concat([tensor_list_size, tf_element_shape], 0)
            tensor = tf.broadcast_to(tf.constant(0.0, dtype=tf.float32), tensor_shape)
            tensor_list = tf.raw_ops.TensorListFromTensor(tensor=tensor,
                                                          element_shape=tf_element_shape)
            tf.raw_ops.TensorListLength(input_handle=tensor_list)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_tensor_list_length_empty_list = [
        dict(tensor_list_size=0, element_shape=[]),
        dict(tensor_list_size=0, element_shape=[2, 3]),
    ]

    @pytest.mark.parametrize("params", test_data_tensor_list_length_empty_list)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_tensor_list_length_empty_list(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_legacy_frontend):
        self._test(*self.create_tensor_list_length_empty_list(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
