# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()

OPS = {
    'tf.raw_ops.AssignAdd': tf.raw_ops.AssignAdd,
    'tf.raw_ops.AssignSub': tf.raw_ops.AssignSub
}


class TestAssignOps(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        if np.issubdtype(self.input_type, np.floating):
            inputs_data['x:0'] = rng.uniform(-2.0, 2.0, x_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            inputs_data['x:0'] = rng.integers(-8, 8, x_shape).astype(self.input_type)
        else:
            inputs_data['x:0'] = rng.integers(0, 8, x_shape).astype(self.input_type)
        return inputs_data

    def create_assign_net(self, const_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, [], 'x')
            variable = tf.raw_ops.VariableV2(shape=const_shape, dtype=input_type)
            if np.issubdtype(self.input_type, np.floating):
                const_value = rng.uniform(-5.0, 5.0, const_shape).astype(self.input_type)
            elif np.issubdtype(self.input_type, np.signedinteger):
                const_value = rng.integers(-8, 8, const_shape).astype(self.input_type)
            else:
                # test bigger unsigned integer constants and avoid overflow by using smaller input values 0..8
                const_value = rng.integers(128, 240, const_shape).astype(self.input_type)
            assign = tf.raw_ops.Assign(ref=variable, value=const_value)
            with tf.control_dependencies([assign]):
                tf.raw_ops.Add(x=x, y=variable)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    def create_assign_op_net(self, const_shape, assign_op):
        self.input_type = np.float32
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(np.float32, [], 'x')
            variable = tf.raw_ops.VariableV2(shape=const_shape, dtype=np.float32)
            const_value = rng.uniform(-2.0, 2.0, const_shape).astype(np.float32)
            const_value2 = rng.uniform(-2.0, 2.0, const_shape).astype(np.float32)
            assign = tf.raw_ops.Assign(ref=variable, value=const_value)
            with tf.control_dependencies([assign]):
                mul1 = tf.raw_ops.Mul(x=variable, y=x, name='mul')
            with tf.control_dependencies([mul1]):
                assign_add = assign_op(ref=assign, value=const_value2)
            with tf.control_dependencies([assign_add]):
                tf.raw_ops.Sub(x=variable, y=mul1, name='sub')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    def create_assign_op_net2(self, const_shape, assign_op):
        self.input_type = np.float32
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(np.float32, [], 'x')
            variable = tf.raw_ops.VariableV2(shape=const_shape, dtype=tf.float32)
            const_value = rng.uniform(-2.0, 2.0, const_shape).astype(np.float32)
            const_value2 = rng.uniform(-2.0, 2.0, const_shape).astype(np.float32)
            assign = tf.raw_ops.Assign(ref=variable, value=const_value)
            assign_add = assign_op(ref=assign, value=const_value2)
            sub = tf.raw_ops.Sub(x=assign_add, y=x, name='sub')
            tf.raw_ops.Mul(x=variable, y=sub, name='mul')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("const_shape", [[], [2], [3, 4], [3, 2, 1, 4]])
    @pytest.mark.parametrize("input_type", [np.int8, np.uint8, np.int16,
                                            np.int32, np.int64,
                                            np.float16, np.float32, np.float64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_assign(self, const_shape, input_type, ie_device, precision, ir_version, temp_dir,
                    use_legacy_frontend):
        if ie_device == 'GPU' and input_type == np.int16:
            pytest.skip("accuracy mismatch for int16 on GPU")
        self._test(*self.create_assign_net(const_shape, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    @pytest.mark.parametrize("const_shape", [[], [2], [3, 4], [3, 2, 1, 4]])
    @pytest.mark.parametrize("assign_op", ['tf.raw_ops.AssignAdd', 'tf.raw_ops.AssignSub'])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_assign_ops(self, const_shape, assign_op, ie_device, precision, ir_version, temp_dir,
                        use_legacy_frontend):
        self._test(*self.create_assign_op_net(const_shape, OPS[assign_op]),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    @pytest.mark.parametrize("const_shape", [[], [2], [3, 4], [3, 2, 1, 4]])
    @pytest.mark.parametrize("assign_op", ['tf.raw_ops.AssignAdd', 'tf.raw_ops.AssignSub'])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_assign_ops2(self, const_shape, assign_op, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        self._test(*self.create_assign_op_net2(const_shape, OPS[assign_op]),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
