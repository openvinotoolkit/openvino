# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()


class TestAssignVariableOps(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = rng.uniform(-2.0, 2.0, x_shape).astype(np.float32)
        return inputs_data

    def create_assign_variable_ops_net(self, const_shape):
        input_type = np.float32
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, [], 'x')
            resource = tf.raw_ops.VarHandleOp(shape=const_shape, dtype=input_type)
            const_value = rng.uniform(-2.0, 2.0, const_shape).astype(input_type)
            assign_variable_op = tf.raw_ops.AssignVariableOp(resource=resource, value=const_value)
            with tf.control_dependencies([assign_variable_op]):
                const_value2 = rng.uniform(-2.0, 2.0, const_shape).astype(input_type)
                assign_add_variable_op = tf.raw_ops.AssignAddVariableOp(resource=resource, value=const_value2)
            with tf.control_dependencies([assign_add_variable_op]):
                const_value3 = rng.uniform(-2.0, 2.0, const_shape).astype(input_type)
                assign_sub_variable_op = tf.raw_ops.AssignSubVariableOp(resource=resource, value=const_value3)
            with tf.control_dependencies([assign_sub_variable_op]):
                resource_value = tf.raw_ops.ReadVariableOp(resource=resource, dtype=tf.float32)
            tf.raw_ops.Mul(x=x, y=resource_value, name='mul')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("const_shape", [[], [2], [3, 4], [3, 2, 1, 4]])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_assign_variable_ops(self, const_shape, ie_device, precision, ir_version, temp_dir,
                                 use_legacy_frontend):
        self._test(*self.create_assign_variable_ops_net(const_shape),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
