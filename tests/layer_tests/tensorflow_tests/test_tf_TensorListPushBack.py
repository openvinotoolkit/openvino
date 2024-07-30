# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from sys import platform


class TestTensorListPushBack(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        assert 'y:0' in inputs_info
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape).astype(self.input_type)
        inputs_data['y:0'] = np.random.randint(-10, 10, y_shape).astype(self.input_type)
        return inputs_data

    def create_tensor_list_push_back(self, element_shape, max_num_elements, element_dtype):
        elem_shape = [2, 3] if element_shape == -1 else element_shape
        self.input_type = element_dtype
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(element_dtype, elem_shape, 'x')
            y = tf.compat.v1.placeholder(element_dtype, elem_shape, 'y')

            empty_tensor_list = tf.raw_ops.EmptyTensorList(element_shape=tf.constant(element_shape, dtype=tf.int32),
                                                           max_num_elements=max_num_elements,
                                                           element_dtype=element_dtype)
            tensor_list1 = tf.raw_ops.TensorListPushBack(input_handle=empty_tensor_list, tensor=x)
            tensor_list2 = tf.raw_ops.TensorListPushBack(input_handle=tensor_list1, tensor=y)
            tf.raw_ops.TensorListStack(input_handle=tensor_list2,
                                       element_shape=tf.constant(element_shape, dtype=tf.int32),
                                       element_dtype=element_dtype)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("element_shape", [[], [2], [3, 4], -1])
    @pytest.mark.parametrize("max_num_elements", [-1, 2, 5])
    @pytest.mark.parametrize("element_dtype", [np.int32, np.float32])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_tensor_list_push_back(self, element_shape, max_num_elements, element_dtype, ie_device, precision,
                                   ir_version, temp_dir,
                                   use_legacy_frontend):
        self._test(*self.create_tensor_list_push_back(element_shape, max_num_elements, element_dtype),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
