# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc, permute_axis
from openvino.tools.mo.ops.op import PermuteAttrs

from unit_tests.utils.graph import build_graph


class Test_TopK(CommonTFLayerTest):
    @staticmethod
    def create_topK_net(shape, k, op_type, ir_version, use_new_frontend):
        """
            Tensorflow net:

                             |-> Values
            Input -> TopK/V2 |
                             |-> Indices


            IR net:

                             |-> Values
            Input -> TopK/V2 |
                             |-> Indices

        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input_tensor = tf.compat.v1.placeholder(tf.float32, shape=shape, name='Input')
            if op_type == 'TopK':
                values, indices = tf.nn.top_k(input=input_tensor, k=k, sorted=True, name='Operation')
            elif op_type == 'TopKV2':
                k_tensor = tf.compat.v1.placeholder(tf.float32, shape=shape, name='Input_k')
                values, indices = tf.raw_ops.TopKV2(input=input_tensor, k=k, sorted=True, name='Operation')
            else:
                raise RuntimeError("Unsupported operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #
        ref_net = None

        if check_ir_version(10, None, ir_version) and not use_new_frontend:
            topk_output_shape = shape.copy()
            inverse_nhwc_nchw = PermuteAttrs.get_nhwc_to_nchw_permutation(len(topk_output_shape)).inv
            topk_axis = permute_axis(len(topk_output_shape) - 1,
                                     inverse_nhwc_nchw)  # we need to permute axis attribute
            topk_output_shape[topk_axis] = k

            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'Const_k_input_data': {'shape': [], 'kind': 'data'},
                'Const_k': {'kind': 'op', 'type': 'Const'},
                'Const_k_data': {'shape': [], 'kind': 'data'},
                'TopK': {'kind': 'op', 'type': 'TopK', 'axis': topk_axis, 'mode': 'max',
                         'sort': 'value'},
                'TopK_data_1': {'shape': topk_output_shape, 'kind': 'data'},
                'TopK_data_2': {'shape': topk_output_shape, 'kind': 'data'},
                'result_1': {'kind': 'op', 'type': 'Result'},
                'result_2': {'kind': 'op', 'type': 'Result'},
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'TopK', {'in': 0}),

                                   ('Const_k_input_data', 'Const_k'),
                                   ('Const_k', 'Const_k_data'),
                                   ('Const_k_data', 'TopK', {'in': 1}),

                                   ('TopK', 'TopK_data_1', {'out': 0}),
                                   ('TopK', 'TopK_data_2', {'out': 1}),
                                   ('TopK_data_1', 'result_1'),
                                   ('TopK_data_2', 'result_2'),
                                   ])

        return tf_net, ref_net

    test_ops = ['TopK', 'TopKV2']

    test_data_1D = [
        dict(shape=[15], k=10),
        dict(shape=[15], k=5),
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.parametrize("op_type", test_ops)
    @pytest.mark.nightly
    def test_TopK_1D(self, params, op_type, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                     use_old_api):
        self._test(*self.create_topK_net(**params, op_type=op_type, ir_version=ir_version,
                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_2D = [
        dict(shape=[14, 15], k=10),
        dict(shape=[14, 15], k=5),
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.parametrize("op_type", test_ops)
    @pytest.mark.nightly
    def test_TopK_2D(self, params, op_type, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                     use_old_api):
        self._test(*self.create_topK_net(**params, op_type=op_type, ir_version=ir_version,
                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_3D = [
        dict(shape=[13, 14, 15], k=10),
        dict(shape=[13, 14, 15], k=5),
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.parametrize("op_type", test_ops)
    @pytest.mark.nightly
    def test_TopK_3D(self, params, op_type, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                     use_old_api):
        self._test(*self.create_topK_net(**params, op_type=op_type, ir_version=ir_version,
                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_4D = [
        dict(shape=[12, 13, 14, 15], k=10),
        dict(shape=[12, 13, 14, 15], k=5),
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.parametrize("op_type", test_ops)
    @pytest.mark.nightly
    def test_TopK_4D(self, params, op_type, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                     use_old_api):
        self._test(*self.create_topK_net(**params, op_type=op_type, ir_version=ir_version,
                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_5D = [
        dict(shape=[11, 12, 13, 14, 15], k=10),
        dict(shape=[11, 12, 13, 14, 15], k=5),
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.parametrize("op_type", test_ops)
    @pytest.mark.nightly
    def test_TopK_5D(self, params, op_type, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                     use_old_api):
        self._test(*self.create_topK_net(**params, op_type=op_type, ir_version=ir_version,
                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
