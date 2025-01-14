# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestTFEye(CommonTFLayerTest):
    eye_output_type_param = np.float32

    # Overload inputs generation to fill dummy Add input with 0
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.zeros(inputs_dict[input]).astype(self.eye_output_type_param)
        return inputs_dict

    def create_tf_eye_net(self, num_rows, num_columns, batch_shape, output_type):
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.global_variables_initializer()
            # batch_shape_input = tf.constant(constant_value)
            if output_type is None:
                eye = tf.eye(num_rows=num_rows, num_columns=num_columns, batch_shape=batch_shape)
            else:
                self.eye_output_type_param = output_type
                eye = tf.eye(num_rows=num_rows, num_columns=num_columns, batch_shape=batch_shape,
                             dtype=tf.as_dtype(output_type))

            # Dummy Add layer to prevent fully const network
            input_zero = tf.compat.v1.placeholder(tf.as_dtype(self.eye_output_type_param), [1], 'Input')
            add = tf.add(eye, input_zero)

            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data = [dict(num_rows=5, num_columns=None, batch_shape=None, output_type=None),
                 dict(num_rows=5, num_columns=5, batch_shape=[2, 3], output_type=np.float32),
                 dict(num_rows=5, num_columns=5, batch_shape=[2, 3], output_type=np.float32),
                 dict(num_rows=5, num_columns=5, batch_shape=[2, 3], output_type=np.float16),
                 dict(num_rows=5, num_columns=5, batch_shape=[2, 3], output_type=np.int32),
                 dict(num_rows=5, num_columns=5, batch_shape=[2, 3], output_type=np.int8),
                 dict(num_rows=8, num_columns=5, batch_shape=None, output_type=np.float32),
                 dict(num_rows=5, num_columns=8, batch_shape=None, output_type=np.float32),
                 dict(num_rows=2, num_columns=2, batch_shape=None, output_type=np.float32),
                 dict(num_rows=6, num_columns=6, batch_shape=[2], output_type=np.float32)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_tf_eye(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if not use_legacy_frontend:
            pytest.xfail(reason="132517: MatrixDiagV3 needs to be supported")
        if ie_device == 'GPU':
            pytest.skip("Roll is not supported on GPU")
        self._test(*self.create_tf_eye_net(**params), ie_device,
                   precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)
