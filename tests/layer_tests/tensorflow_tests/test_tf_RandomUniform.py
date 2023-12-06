# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import tensorflow as tf
from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, connect, \
    shaped_data, connect_front


class TestRandomUniform(CommonTFLayerTest):
    def create_tf_random_uniform_net(self, global_seed, op_seed, x_shape, min_val, max_val,
                                     input_type, precision,
                                     ir_version, use_new_frontend):
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, x_shape, 'Input')
            if global_seed is not None:
                tf.compat.v1.random.set_random_seed(global_seed)
            tf.random.uniform(x_shape, seed=op_seed, dtype=input_type,
                              minval=min_val,
                              maxval=max_val) + x

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        if check_ir_version(10, None, ir_version):
            const_for_layer_tests = lambda name, value, shape, shape1: {
                **{name + '_dd': {'kind': 'data', 'value': value, 'shape': shape1}},
                **{name: {'kind': 'op', 'type': 'Const'}},
                **shaped_data(name + '_d', shape)}

            connect_const_for_layer_tests = lambda first_tensor_name, second_tensor_name: [
                *connect_front(first_tensor_name + '_dd', first_tensor_name),
                *connect(first_tensor_name, second_tensor_name)]

            nodes_attributes = {
                **regular_op_with_shaped_data('input', x_shape, {'type': 'Parameter'}),
                **const_for_layer_tests('shape', x_shape, int64_array([len(x_shape)]),
                                        int64_array([len(x_shape)])),
                **const_for_layer_tests('min_val', min_val, int64_array([]), int64_array([1])),
                **const_for_layer_tests('max_val', max_val, int64_array([]), int64_array([1])),
                **regular_op_with_shaped_data('random_uniform', x_shape, {'type': 'RandomUniform'}),
                **regular_op_with_shaped_data('convert', x_shape, {'type': 'Convert'}),
                **regular_op_with_shaped_data('add', x_shape, {'type': 'Add'}),
                **regular_op_with_shaped_data('result', x_shape, {'type': 'Result'}),

            }

            if precision == 'FP16' and input_type == tf.float32:
                ref_net = build_graph(nodes_attributes,
                                      [*connect_const_for_layer_tests('shape', '0:random_uniform'),
                                       *connect_const_for_layer_tests('min_val',
                                                                      '1:random_uniform'),
                                       *connect_const_for_layer_tests('max_val',
                                                                      '2:random_uniform'),
                                       *connect('random_uniform', 'convert'),
                                       *connect('convert', '0:add'),
                                       *connect('input', '1:add'),
                                       *connect('add', 'result')])
            else:
                ref_net = build_graph(nodes_attributes,
                                      [*connect_const_for_layer_tests('shape', '0:random_uniform'),
                                       *connect_const_for_layer_tests('min_val',
                                                                      '1:random_uniform'),
                                       *connect_const_for_layer_tests('max_val',
                                                                      '2:random_uniform'),
                                       *connect('random_uniform', '0:add'),
                                       *connect('input', '1:add'),
                                       *connect('add', 'result')])

        return tf_net, ref_net

    test_data_basic = [
        dict(global_seed=32465, op_seed=48971, min_val=0.0, max_val=1.0, x_shape=[3, 7],
             input_type=tf.float32),
        dict(global_seed=78132, op_seed=None, min_val=-200, max_val=-50, x_shape=[5, 8],
             input_type=tf.int32)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_tf_fe
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_random_uniform_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_new_frontend, use_old_api):
        if ie_device == 'GPU':
            pytest.skip("RandomUniform is not supported on GPU")
        self._test(
            *self.create_tf_random_uniform_net(**params, precision=precision, ir_version=ir_version,
                                               use_new_frontend=use_new_frontend), ie_device,
            precision, temp_dir=temp_dir, ir_version=ir_version, use_new_frontend=use_new_frontend,
            use_old_api=use_old_api, **params)

    test_data_other = [
        dict(global_seed=None, op_seed=56197, min_val=-100, max_val=100, x_shape=[6],
             input_type=tf.float32),
        dict(global_seed=None, op_seed=56197, min_val=-100, max_val=100, x_shape=[1, 2, 1, 1],
             input_type=tf.float32),
        dict(global_seed=4571, op_seed=48971, min_val=1.5, max_val=2.3, x_shape=[7],
             input_type=tf.float32),
        dict(global_seed=32465, op_seed=12335, min_val=-150, max_val=-100, x_shape=[18],
             input_type=tf.int32)]

    @pytest.mark.parametrize("params", test_data_other)
    @pytest.mark.nightly
    def test_random_uniform_other(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_new_frontend, use_old_api):
        if ie_device == 'GPU':
            pytest.skip("RandomUniform is not supported on GPU")
        self._test(
            *self.create_tf_random_uniform_net(**params, precision=precision, ir_version=ir_version,
                                               use_new_frontend=use_new_frontend), ie_device,
            precision, temp_dir=temp_dir, ir_version=ir_version, use_new_frontend=use_new_frontend,
            use_old_api=use_old_api, **params)
