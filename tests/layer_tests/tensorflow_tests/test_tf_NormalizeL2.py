# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc
from openvino.tools.mo.front.common.partial_infer.utils import int64_array

from unit_tests.utils.graph import build_graph


class TestNormalizeL2(CommonTFLayerTest):
    @staticmethod
    def build_tf_graph(shape, axes):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            # Permute NCHW -> NHWC for TF network creation
            net_shape = permute_nchw_to_nhwc(shape)

            data = tf.compat.v1.placeholder(tf.float32, shape=net_shape, name='data')

            result = tf.math.l2_normalize(data,
                                          axes,
                                          name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def
            return tf_net

    @staticmethod
    def create_normalize_l2_net_fusable(shape, axes, output_axes, ir_version, use_new_frontend):
        tf_net = TestNormalizeL2.build_tf_graph(shape, axes)

        nodes_attributes = {
            'input': {'kind': 'op', 'type': 'Parameter'},
            'input_data': {'shape': shape, 'kind': 'data'},
            'axes_input_data': {'shape': int64_array([len(axes)]), 'kind': 'data',
                                'value': int64_array(output_axes)},
            'axes': {'kind': 'op', 'type': 'Const'},
            'axes_data': {'shape': int64_array([len(axes)]), 'kind': 'data'},
            'normalize_l2': {'kind': 'op', 'type': 'NormalizeL2'},
            'normalize_l2_data': {'shape': shape, 'kind': 'data'},
            'result': {'kind': 'op', 'type': 'Result'},
        }

        ref_net = build_graph(nodes_attributes,
                              [('input', 'input_data'),
                               ('input_data', 'normalize_l2', {'out': 0, 'in': 0}),
                               ('axes_input_data', 'axes'),
                               ('axes', 'axes_data'),
                               ('axes_data', 'normalize_l2', {'in': 1, 'out': 0}),
                               ('normalize_l2', 'normalize_l2_data'),
                               ('normalize_l2_data', 'result'),
                               ])

        if use_new_frontend:
            ref_net = None
        return tf_net, ref_net

    @staticmethod
    def create_normalize_l2_net_non_fusable(shape, axes, output_axes, ir_version, use_new_frontend):
        tf_net = TestNormalizeL2.build_tf_graph(shape, axes)

        reduced_shape = permute_nchw_to_nhwc(shape).copy()
        for axis in axes:
            reduced_shape[axis] = 1
        reduced_shape = permute_nchw_to_nhwc(reduced_shape)

        eltwise_shapes = int64_array(np.ones(len(shape)))
        nodes_attributes = {
            'input': {'kind': 'op', 'type': 'Parameter'},
            'input_data': {'shape': shape, 'kind': 'data'},

            'power_const_input_data': {'shape': int64_array([1]), 'kind': 'data',
                                       'value': np.array([2.0])},
            'power_const': {'kind': 'op', 'type': 'Const'},
            'power_const_data': {'shape': eltwise_shapes, 'kind': 'data'},
            'power': {'kind': 'op', 'type': 'Power'},
            'power_data': {'shape': shape, 'kind': 'data'},

            'reduce': {'kind': 'op', 'type': 'ReduceSum', 'keep_dims': True},
            'reduce_data': {'shape': reduced_shape, 'kind': 'data'},
            'reduce_axes_input_data': {'shape': int64_array([len(axes)]), 'kind': 'data',
                                       'value': int64_array(output_axes)},
            'reduce_axes': {'kind': 'op', 'type': 'Const'},
            'reduce_axes_data': {'shape': int64_array([len(axes)]), 'kind': 'data'},

            'maximum_const_input_data': {'shape': int64_array([1]), 'kind': 'data',
                                         'value': np.array([1e-12])},
            'maximum_const': {'kind': 'op', 'type': 'Const'},
            'maximum_const_data': {'shape': eltwise_shapes, 'kind': 'data'},
            'maximum': {'kind': 'op', 'type': 'Maximum'},
            'maximum_data': {'shape': reduced_shape, 'kind': 'data'},

            'power2_const_input_data': {'shape': int64_array([1]), 'kind': 'data',
                                        'value': np.array([-0.5])},
            'power2_const': {'kind': 'op', 'type': 'Const'},
            'power2_const_data': {'shape': eltwise_shapes, 'kind': 'data'},
            'power2': {'kind': 'op', 'type': 'Power'},
            'power2_data': {'shape': reduced_shape, 'kind': 'data'},

            'multiply': {'kind': 'op', 'type': 'Multiply'},
            'multiply_data': {'shape': shape, 'kind': 'data'},
            'result': {'kind': 'op', 'type': 'Result'},
        }

        ref_net = build_graph(nodes_attributes,
                              [('input', 'input_data'),

                               ('input_data', 'power', {'out': 0, 'in': 0}),
                               ('power_const_input_data', 'power_const'),
                               ('power_const', 'power_const_data'),
                               ('power_const_data', 'power', {'out': 0, 'in': 1}),
                               ('power', 'power_data'),

                               ('power_data', 'reduce', {'out': 0, 'in': 0}),
                               ('reduce_axes_input_data', 'reduce_axes'),
                               ('reduce_axes', 'reduce_axes_data'),
                               ('reduce_axes_data', 'reduce', {'out': 0, 'in': 1}),
                               ('reduce', 'reduce_data'),

                               ('reduce_data', 'maximum', {'out': 0, 'in': 0}),
                               ('maximum_const_input_data', 'maximum_const'),
                               ('maximum_const', 'maximum_const_data'),
                               ('maximum_const_data', 'maximum', {'out': 0, 'in': 1}),
                               ('maximum', 'maximum_data'),

                               ('maximum_data', 'power2', {'out': 0, 'in': 0}),
                               ('power2_const_input_data', 'power2_const'),
                               ('power2_const', 'power2_const_data'),
                               ('power2_const_data', 'power2', {'out': 0, 'in': 1}),
                               ('power2', 'power2_data'),

                               ('input_data', 'multiply', {'out': 0, 'in': 0}),
                               ('power2_data', 'multiply', {'out': 0, 'in': 1}),
                               ('multiply', 'multiply_data'),
                               ('multiply_data', 'result'),
                               ])

        if use_new_frontend:
            ref_net = None
        return tf_net, ref_net

    test_data_fusable_precommit = [
        pytest.param(dict(shape=[2, 3, 5], axes=[1, -1], output_axes=[1, 2]),
                     marks=pytest.mark.skip(reason="Skipped until fixed")),
        pytest.param(dict(shape=[2, 3, 5, 7], axes=[1, 2, 3], output_axes=[2, 3, 1]),
                     marks=pytest.mark.skip(reason="Skipped until fixed"))
    ]

    @pytest.mark.parametrize("params", test_data_fusable_precommit)
    @pytest.mark.precommit
    def test_NormalizeL2_fusable_precommit(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_new_frontend, use_old_api):
        self._test(*self.create_normalize_l2_net_fusable(**params, ir_version=ir_version,
                                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_non_fusable_precommit = [
        pytest.param(dict(shape=[2, 3, 5], axes=[0, 1, 2], output_axes=[0, 1, 2]),
                     marks=pytest.mark.skip(reason="Skipped until fixed")),
        pytest.param(dict(shape=[2, 3, 5, 7, 9], axes=[-1], output_axes=[1]),
                     marks=pytest.mark.skip(reason="Skipped until fixed")),
        pytest.param(dict(shape=[2, 3, 5, 7, 9], axes=[1, 2, 3, 4], output_axes=[2, 3, 4, 1]),
                     marks=pytest.mark.skip(reason="Skipped until fixed"))
    ]

    @pytest.mark.parametrize("params", test_data_non_fusable_precommit)
    @pytest.mark.precommit
    def test_NormalizeL2_non_fusable_precommit(self, params, ie_device, precision, ir_version,
                                               temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_normalize_l2_net_non_fusable(**params, ir_version=ir_version,
                                                             use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_fusable = [
        dict(shape=[5, 6], axes=[1], output_axes=[1]),
        dict(shape=[2, 3, 5], axes=[1], output_axes=[1]),
        dict(shape=[2, 3, 5], axes=[-2], output_axes=[1]),
        pytest.param(dict(shape=[2, 3, 5], axes=[1, -1], output_axes=[1, 2]), marks=pytest.mark.precommit_tf_fe),
        dict(shape=[2, 3, 5, 7], axes=[-1], output_axes=[1]),
        dict(shape=[2, 3, 5, 7], axes=[1, 2, 3], output_axes=[2, 3, 1]),
    ]

    @pytest.mark.parametrize("params", test_data_fusable)
    @pytest.mark.nightly
    def test_NormalizeL2_fusable(self, params, ie_device, precision, ir_version, temp_dir,
                                 use_new_frontend, use_old_api):
        self._test(*self.create_normalize_l2_net_fusable(**params, ir_version=ir_version,
                                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_non_fusable = [
        dict(shape=[5], axes=[0], output_axes=[0]),
        dict(shape=[5, 6], axes=[0], output_axes=[0]),
        dict(shape=[5, 6], axes=[0, 1], output_axes=[0, 1]),
        dict(shape=[2, 3, 5], axes=[0], output_axes=[0]),
        dict(shape=[2, 3, 5], axes=[2], output_axes=[2]),
        dict(shape=[2, 3, 5], axes=[0, 1, 2], output_axes=[0, 1, 2]),
        dict(shape=[2, 3, 5, 7], axes=[0], output_axes=[0]),
        dict(shape=[2, 3, 5, 7], axes=[1], output_axes=[2]),
        dict(shape=[2, 3, 5, 7], axes=[2], output_axes=[3]),
        dict(shape=[2, 3, 5, 7], axes=[1, 2], output_axes=[2, 3]),
        dict(shape=[2, 3, 5, 7], axes=[1, 3], output_axes=[2, 1]),
        dict(shape=[2, 3, 5, 7], axes=[0, 1, 2], output_axes=[0, 2, 3]),
        dict(shape=[2, 3, 5, 7, 9], axes=[-1], output_axes=[1]),
        dict(shape=[2, 3, 5, 7, 9], axes=[1, 2, 3, 4], output_axes=[2, 3, 4, 1]),
    ]

    @pytest.mark.parametrize("params", test_data_non_fusable)
    @pytest.mark.nightly
    def test_NormalizeL2_non_fusable(self, params, ie_device, precision, ir_version, temp_dir,
                                     use_new_frontend, use_old_api):
        self._test(*self.create_normalize_l2_net_non_fusable(**params, ir_version=ir_version,
                                                             use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_new_frontend=use_new_frontend, use_old_api=use_old_api)
