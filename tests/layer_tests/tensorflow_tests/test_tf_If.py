# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestIfFloat(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'cond:0' in inputs_info, "Test error: inputs_info must contain `cond`"
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
        cond_shape = inputs_info['cond:0']
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['cond:0'] = np.random.randint(0, 2, cond_shape).astype(bool)
        inputs_data['x:0'] = np.random.randint(1, 10, x_shape).astype(np.float32)
        inputs_data['y:0'] = np.random.randint(-50, 50, y_shape).astype(np.float32)
        return inputs_data

    def create_if_net(self, x_shape, y_shape, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def if_function(cond, x, y):
            def then_branch():
                output1 = tf.add(x, y)
                output2 = tf.multiply(x, y)
                output3 = tf.subtract(x, y)
                return output1, output2, output3

            def else_branch():
                const_two = tf.constant(2.0, dtype=tf.float32)
                output1 = tf.add(y, const_two)
                output2 = tf.multiply(const_two, y)
                output3 = x - y + const_two
                return output1, output2, output3

            if_output = tf.cond(cond, then_branch, else_branch)
            output1 = tf.identity(if_output[0], name='output1')
            output2 = tf.identity(if_output[1], name='output2')
            output3 = tf.identity(if_output[2], name='output3')
            return output1, output2, output3

        tf_if_graph = tf.function(if_function)
        cond = np.random.randint(0, 2, []).astype(bool)
        x = np.random.randint(1, 10, x_shape).astype(np.float32)
        y = np.random.randint(-50, 50, y_shape).astype(np.float32)
        concrete_func = tf_if_graph.get_concrete_function(cond, x, y)

        # lower_control_flow defines representation of If operation
        # in case of lower_control_flow=True it is decomposed into Switch and Merge nodes
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        tf_net = frozen_func.graph.as_graph_def(add_shapes=True)
        return tf_net, None

    test_data_basic = [
        dict(x_shape=[3], y_shape=[2, 3], lower_control_flow=False),
        dict(x_shape=[3], y_shape=[2, 3], lower_control_flow=True),
        dict(x_shape=[2, 1, 4], y_shape=[2, 1, 4], lower_control_flow=False),
        dict(x_shape=[2, 1, 4], y_shape=[2, 1, 4], lower_control_flow=True)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_if_basic(self, params, ie_device, precision, ir_version, temp_dir,
                      use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.xfail('104855')
        self._test(*self.create_if_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestIfInt(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'cond:0' in inputs_info, "Test error: inputs_info must contain `cond`"
        assert 'ind:0' in inputs_info, "Test error: inputs_info must contain `ind`"
        assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
        cond_shape = inputs_info['cond:0']
        ind_shape = inputs_info['ind:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['cond:0'] = np.random.randint(0, 2, cond_shape).astype(bool)
        inputs_data['ind:0'] = np.random.randint(1, 10, ind_shape).astype(np.int32)
        inputs_data['y:0'] = np.random.randint(-50, 50, y_shape).astype(np.float32)
        return inputs_data

    def create_if_net(self, ind_shape, y_shape, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def if_function(cond, ind, y):
            def then_branch():
                const_one = tf.constant(1, dtype=tf.int32)
                output1 = tf.add(ind, const_one)
                output2 = tf.multiply(tf.cast(output1, tf.float32), y)
                output3 = tf.subtract(tf.cast(output1, tf.float32), y)
                return output1, output2, output3

            def else_branch():
                const_two = tf.constant(2, dtype=tf.int32)
                output1 = tf.add(ind, const_two)
                output2 = tf.multiply(tf.cast(output1, tf.float32), y)
                output3 = tf.cast(output1, tf.float32) - y
                return output1, output2, output3

            if_output = tf.cond(cond, then_branch, else_branch)
            output1 = tf.identity(if_output[0], name='output1')
            output2 = tf.identity(if_output[1], name='output2')
            output3 = tf.identity(if_output[2], name='output3')
            return output1, output2, output3

        tf_if_graph = tf.function(if_function)
        cond = np.random.randint(0, 2, []).astype(bool)
        ind = np.random.randint(1, 10, ind_shape).astype(np.int32)
        y = np.random.randint(-50, 50, y_shape).astype(np.float32)
        concrete_func = tf_if_graph.get_concrete_function(cond, ind, y)

        # lower_control_flow defines representation of If operation
        # in case of lower_control_flow=True it is decomposed into Switch and Merge nodes
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        return graph_def, None

    test_data_basic = [
        dict(ind_shape=[3], y_shape=[2, 3], lower_control_flow=False),
        dict(ind_shape=[3], y_shape=[2, 3], lower_control_flow=True),
        dict(ind_shape=[2, 1, 4], y_shape=[2, 1, 4], lower_control_flow=False),
        dict(ind_shape=[2, 1, 4], y_shape=[2, 1, 4], lower_control_flow=True)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_if_basic(self, params, ie_device, precision, ir_version, temp_dir,
                      use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.xfail('104855')
        self._test(*self.create_if_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestNestedIf(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `cond`"
        assert 'y:0' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'z:0' in inputs_info, "Test error: inputs_info must contain `y`"
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        z_shape = inputs_info['z:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(0, 6, x_shape).astype(np.int32)
        inputs_data['y:0'] = np.random.randint(1, 10, y_shape).astype(np.float32)
        inputs_data['z:0'] = np.random.randint(-50, 50, z_shape).astype(np.float32)
        return inputs_data

    def create_if_net(self, y_shape, z_shape, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def if_function(x, y, z):
            def nested_then_branch():
                add = tf.add(y, z)
                return add

            def nested_else_branch():
                mul = tf.multiply(y, z)
                return mul

            def then_branch():
                output1 = tf.cond(x > 4, nested_then_branch, nested_else_branch)
                output2 = tf.multiply(y, z)
                output3 = tf.subtract(y, z)
                return output1, output2, output3

            def else_branch():
                const_two = tf.constant(2.0, dtype=tf.float32)
                output1 = tf.add(y, const_two)
                output1 = tf.add(output1, z)
                output2 = tf.multiply(z, y)
                output3 = z - y + const_two
                return output1, output2, output3

            if_output = tf.cond(x < 2, then_branch, else_branch)
            output1 = tf.identity(if_output[0], name='output1')
            output2 = tf.identity(if_output[1], name='output2')
            output3 = tf.identity(if_output[2], name='output3')
            return output1, output2, output3

        tf_if_graph = tf.function(if_function)
        x = np.random.randint(0, 8, []).astype(np.int32)
        y = np.random.randint(1, 10, y_shape).astype(np.float32)
        z = np.random.randint(-50, 50, z_shape).astype(np.float32)
        concrete_func = tf_if_graph.get_concrete_function(x, y, z)

        # lower_control_flow defines representation of If operation
        # in case of lower_control_flow=True it is decomposed into Switch and Merge nodes
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        tf_net = frozen_func.graph.as_graph_def(add_shapes=True)
        return tf_net, None

    test_data_basic = [
        dict(y_shape=[3], z_shape=[2, 3], lower_control_flow=False),
        dict(y_shape=[3], z_shape=[2, 3], lower_control_flow=True),
        dict(y_shape=[2, 1, 4], z_shape=[2, 1, 4], lower_control_flow=False),
        dict(y_shape=[2, 1, 4], z_shape=[2, 1, 4], lower_control_flow=True)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_if_basic(self, params, ie_device, precision, ir_version, temp_dir,
                      use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.xfail('104855')
        self._test(*self.create_if_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestSequantialIfs(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'cond:0' in inputs_info, "Test error: inputs_info must contain `cond`"
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
        cond_shape = inputs_info['cond:0']
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['cond:0'] = np.random.randint(0, 2, cond_shape).astype(bool)
        inputs_data['x:0'] = np.random.randint(1, 10, x_shape).astype(np.float32)
        inputs_data['y:0'] = np.random.randint(-50, 50, y_shape).astype(np.float32)
        return inputs_data

    def create_sequential_ifs_net(self, x_shape, y_shape, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def sequential_ifs(cond, x, y):
            def if1(cond1, x1, y1):
                def then_branch():
                    add = tf.add(x1, y1)
                    mul = tf.multiply(x1, y1)
                    return add, mul

                def else_branch():
                    const_two = tf.constant(2.0, dtype=tf.float32)
                    add = tf.add(y1, const_two)
                    mul = tf.multiply(const_two, y1)
                    return add, mul

                if1_op = tf.cond(cond1, then_branch, else_branch)
                output1 = tf.identity(if1_op[0], name='output1')
                output2 = tf.identity(if1_op[1], name='output2')
                return output1, output2

            def if2(cond1, x2, y2):
                def then_branch():
                    const_two = tf.constant(2.0, dtype=tf.float32)
                    add = tf.add(y2, const_two)
                    mul = tf.multiply(const_two, y2)
                    return add, mul

                def else_branch():
                    add = tf.add(x2, y2)
                    mul = tf.multiply(x2, y2)
                    return add, mul

                if2_op = tf.cond(cond1, then_branch, else_branch)
                output1 = tf.identity(if2_op[0], name='output1')
                output2 = tf.identity(if2_op[1], name='output2')
                return output1, output2

            output1, output2 = if1(cond, x, y)
            const_ten = tf.constant(10.0, dtype=tf.float32)
            output1 = tf.add(output1, const_ten)
            output1, output2 = if2(cond, output1, output2)
            return output1, output2

        tf_if_graph = tf.function(sequential_ifs)
        cond = np.random.randint(0, 2, []).astype(bool)
        x = np.random.randint(1, 10, x_shape).astype(np.float32)
        y = np.random.randint(-50, 50, y_shape).astype(np.float32)
        concrete_func = tf_if_graph.get_concrete_function(cond, x, y)

        # lower_control_flow defines representation of If operation
        # in case of lower_control_flow=True it is decomposed into Switch and Merge nodes
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        tf_net = frozen_func.graph.as_graph_def(add_shapes=True)
        return tf_net, None

    test_data_basic = [
        dict(x_shape=[3], y_shape=[2, 3], lower_control_flow=False),
        dict(x_shape=[3], y_shape=[2, 3], lower_control_flow=True),
        dict(x_shape=[2, 1, 4], y_shape=[2, 1, 4], lower_control_flow=False),
        dict(x_shape=[2, 1, 4], y_shape=[2, 1, 4], lower_control_flow=True)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_if_basic(self, params, ie_device, precision, ir_version, temp_dir,
                      use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.xfail('104855')
        self._test(*self.create_sequential_ifs_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
