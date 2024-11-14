# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from sys import platform


class TestTensorListConcatV2(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape).astype(self.input_type)
        return inputs_data

    def create_tensor_list_concat_v2(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            # set defined dimensions for placeholder creation
            place_shape = [3 if i < 0 else i for i in input_shape]
            x = tf.compat.v1.placeholder(input_type, place_shape, 'x')
            tensor_list = tf.raw_ops.TensorListFromTensor(tensor=x,
                                                          element_shape=tf.constant(input_shape[1:], dtype=tf.int32))
            tf.raw_ops.TensorListConcatV2(input_handle=tensor_list,
                                          element_shape=tf.constant(input_shape[1:], dtype=tf.int32),
                                          element_dtype=input_type,
                                          leading_dims=tf.constant([], dtype=tf.int64))
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("input_shape", [[10, 20], [2, 3, 4], [3, -1], [2, -1, -1, -1]])
    @pytest.mark.parametrize("input_type", [np.float32, np.int32])
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_tensor_list_concat_v2(self, input_shape, input_type, ie_device, precision, ir_version, temp_dir,
                                   use_legacy_frontend):
        self._test(*self.create_tensor_list_concat_v2(input_shape, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
