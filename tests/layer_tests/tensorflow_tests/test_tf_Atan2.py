# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestAtan2(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'y:0' in inputs_info
        assert 'x:0' in inputs_info
        y_shape = inputs_info['y:0']
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['y:0'] = np.random.rand(*y_shape).astype(self.input_type) - np.random.rand(*y_shape).astype(self.input_type)
        inputs_data['x:0'] = np.random.rand(*x_shape).astype(self.input_type) - np.random.rand(*x_shape).astype(self.input_type)
        return inputs_data

    def create_atan2_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            y = tf.compat.v1.placeholder(input_type, input_shape, 'y')
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            tf.raw_ops.Atan2(y=y, x=x)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[1, 2], input_type=np.float32),
        dict(input_shape=[2, 3, 4], input_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_atan2_basic(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_atan2_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)


class TestAtan2ZeroEdgeCases(CommonTFLayerTest):
    """Test atan2 with zero in the inputs - regression test for atan2(0, 0)
    returning NaN.

    The common_translators atan2 decomposition computes atan(y/x), which for
    x=0, y=0 produces atan(NaN) = NaN. Test handling for all IEEE 754
    signed-zero cases:
      atan2(+0, +0) = +0
      atan2(-0, +0) = -0
      atan2(+0, -0) = +π
      atan2(-0, -0) = -π
    """

    def _prepare_input(self, inputs_info):
        assert 'y:0' in inputs_info
        assert 'x:0' in inputs_info
        if self.zero_case == "both_positive_zero":
            y = np.zeros(inputs_info['y:0'], dtype=np.float32)
            x = np.zeros(inputs_info['x:0'], dtype=np.float32)
        elif self.zero_case == "mixed_zeros_and_axes":
            y = np.array([0.0, 1.0, 0.0, -1.0, 0.0], dtype=np.float32)
            x = np.array([0.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        elif self.zero_case == "all_zeros":
            y = np.zeros(inputs_info['y:0'], dtype=np.float32)
            x = np.zeros(inputs_info['x:0'], dtype=np.float32)
        elif self.zero_case == "x_zero_various_y":
            y = np.array([0.0, 1.0, -1.0], dtype=np.float32)
            x = np.zeros(inputs_info['x:0'], dtype=np.float32)
        elif self.zero_case == "signed_neg_zero_x":
            y = np.array([0.0, -0.0], dtype=np.float32)
            x = np.array([-0.0, -0.0], dtype=np.float32)
        elif self.zero_case == "signed_neg_zero_y":
            y = np.array([-0.0, -0.0], dtype=np.float32)
            x = np.array([0.0, -0.0], dtype=np.float32)
        elif self.zero_case == "all_four_signed_zero_combos":
            y = np.array([0.0, -0.0, 0.0, -0.0], dtype=np.float32)
            x = np.array([0.0, 0.0, -0.0, -0.0], dtype=np.float32)
        return {'y:0': y, 'x:0': x}

    def create_atan2_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            y = tf.compat.v1.placeholder(input_type, input_shape, 'y')
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            atan2_res = tf.raw_ops.Atan2(y=y, x=x)
            # Make signed-zero outputs observable: +0 -> +1, -0 -> -1.
            # Wrap in where + sign to keep only finite values in the output:
            #   sign(1/+0) = sign(+inf) = +1
            #   sign(1/-0) = sign(-inf) = -1
            #   non-zero outputs are kept as-is.
            zero_const = tf.constant(0.0, dtype=input_type)
            is_zero = tf.equal(atan2_res, zero_const)
            sign_of_recip = tf.sign(tf.math.reciprocal(atan2_res))
            tf.where(is_zero, sign_of_recip, atan2_res,
                     name="atan2_sign_aware")
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    _case_shapes = {
        "both_positive_zero": [1],
        "mixed_zeros_and_axes": [5],
        "all_zeros": [3],
        "x_zero_various_y": [3],
        "signed_neg_zero_x": [2],
        "signed_neg_zero_y": [2],
        "all_four_signed_zero_combos": [4],
    }

    @pytest.mark.parametrize("zero_case", [
        "both_positive_zero",
        "mixed_zeros_and_axes",
        "all_zeros",
        "x_zero_various_y",
        "signed_neg_zero_x",
        "signed_neg_zero_y",
        "all_four_signed_zero_combos",
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_atan2_zero_edge_cases(self, ie_device, precision, ir_version,
                                   temp_dir, zero_case):
        self.zero_case = zero_case
        input_shape = self._case_shapes[zero_case]
        self._test(*self.create_atan2_net(input_shape, np.float32),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
