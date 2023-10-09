# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestTensorArraySizeV3(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data' in inputs_info
        assert 'indices' in inputs_info
        data_shape = inputs_info['data']
        inputs_data = {}
        inputs_data['data'] = np.random.randint(-10, 10, data_shape).astype(self.data_type)
        rng = np.random.default_rng()
        inputs_data['indices'] = rng.permutation(self.size).astype(np.int32)
        return inputs_data

    def create_tensor_array_size_v3(self, data_shape, data_type):
        size = data_shape[0]
        self.data_type = data_type
        self.size = size
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(data_type, data_shape, 'data')
            indices = tf.compat.v1.placeholder(tf.int32, [size], 'indices')
            size_const = tf.constant(size, dtype=tf.int32, shape=[])
            handle, flow = tf.raw_ops.TensorArrayV3(size=size_const, dtype=tf.as_dtype(data_type))
            flow = tf.raw_ops.TensorArrayScatterV3(handle=handle, indices=indices, value=data, flow_in=flow)
            tf.raw_ops.TensorArraySizeV3(handle=handle, flow_in=flow)
            tf.raw_ops.TensorArrayCloseV3(handle=handle)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(data_shape=[5], data_type=np.float32),
        dict(data_shape=[10, 20, 30], data_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_tensor_array_size_v3_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                        use_new_frontend, use_old_api):
        self._test(*self.create_tensor_array_size_v3(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
