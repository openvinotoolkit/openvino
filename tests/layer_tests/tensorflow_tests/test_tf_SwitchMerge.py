# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSwitchMerge(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        x_shape = inputs_info['x:0']
        inputs_data = {}
        rng = np.random.default_rng()
        inputs_data['x:0'] = rng.integers(-10, 10, x_shape).astype(self.x_type)
        return inputs_data

    def merge_eliminating_several_cond_flows_net(self, x_shape, x_type, cond_value):
        self.x_type = x_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(x_type, x_shape, 'x')
            cond = tf.constant(cond_value, dtype=tf.bool)
            switch_false, switch_true = tf.raw_ops.Switch(data=x, pred=cond)

            cond2 = tf.constant(cond_value, dtype=tf.bool)
            switch2_false, switch2_true = tf.raw_ops.Switch(data=cond2, pred=cond2)
            with tf.control_dependencies([switch2_true]):
                const_sub = tf.constant(5, dtype=x_type)
            with tf.control_dependencies([switch2_false]):
                const_add = tf.constant(2, dtype=x_type)

            add = tf.raw_ops.AddV2(x=switch_false, y=const_add)
            sub = tf.raw_ops.Sub(x=switch_true, y=const_sub)
            merge = tf.raw_ops.Merge(inputs=[add, sub])
            const_main = tf.constant(1, dtype=x_type)
            tf.raw_ops.AddV2(x=merge[0], y=const_main)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[]),
        dict(x_shape=[2]),
        dict(x_shape=[4, 3]),
    ]

    @pytest.mark.parametrize("cond_value", [
        True, False
    ])
    @pytest.mark.parametrize("x_type", [
        np.float32, np.int32
    ])
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_merge_eliminating_several_cond_flows(self, params, cond_value, x_type, ie_device, precision, ir_version,
                                                  temp_dir,
                                                  use_legacy_frontend):
        self._test(*self.merge_eliminating_several_cond_flows_net(**params, cond_value=cond_value, x_type=x_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestSwitchMergeWithVariablePredicate(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        x_shape = inputs_info['x:0']
        inputs_data = {}
        rng = np.random.default_rng()
        inputs_data['x:0'] = rng.integers(-10, 10, x_shape).astype(np.float32)
        inputs_data['cond:0'] = np.array(self.cond_value, dtype=bool)
        return inputs_data

    def switch_merge_with_variable_predicate_net(self, x_shape, cond_shape, cond_value):
        self.cond_value = cond_value
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'x')
            cond = tf.compat.v1.placeholder(tf.bool, cond_shape, 'cond')
            const_add = tf.constant(3, dtype=tf.float32)
            const_sub = tf.constant(1, dtype=tf.float32)
            switch_false, switch_true = tf.raw_ops.Switch(data=x, pred=cond)
            add = tf.raw_ops.AddV2(x=switch_false, y=const_add)
            sub = tf.raw_ops.Sub(x=switch_true, y=const_sub)
            merge = tf.raw_ops.Merge(inputs=[add, sub])
            const_main = tf.constant(1, dtype=tf.float32)
            tf.raw_ops.AddV2(x=merge[0], y=const_main, name='add_res')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('x_shape', [[], [2], [3, 2]])
    @pytest.mark.parametrize('cond_shape', [None, []])
    @pytest.mark.parametrize('cond_value', [True, False])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_switch_merge_with_variable_predicate(self, x_shape, cond_shape, cond_value,
                                                  ie_device, precision, ir_version, temp_dir,
                                                  use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("156244: accuracy error on GPU")
        self._test(*self.switch_merge_with_variable_predicate_net(x_shape, cond_shape, cond_value),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
