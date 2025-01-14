# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestWhere(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(0, 2, inputs_dict[input]).astype(bool)
        return inputs_dict

    def create_net(self, condition_shape, shape_than, else_shape, ir_version):
        """
            ONNX net                                  IR net

            Input->Where->Output   =>    Input->Select
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        input_cond = helper.make_tensor_value_info('input_cond', TensorProto.BOOL, condition_shape)
        input_than = helper.make_tensor_value_info('input_than', TensorProto.BOOL, shape_than)
        input_else = helper.make_tensor_value_info('input_else', TensorProto.BOOL, else_shape)
        output = helper.make_tensor_value_info('output', TensorProto.BOOL, condition_shape)

        node_def = helper.make_node(
            'Where',
            inputs=['input_cond', 'input_than', 'input_else'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [input_cond, input_than, input_else],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #   Create reference IR net

        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input_cond': {'kind': 'op', 'type': 'Parameter'},
                'input_cond_data': {'shape': condition_shape, 'kind': 'data'},

                'input_than': {'kind': 'op', 'type': 'Parameter'},
                'input_than_data': {'shape': shape_than, 'kind': 'data'},

                'input_else': {'kind': 'op', 'type': 'Parameter'},
                'input_else_data': {'shape': else_shape, 'kind': 'data'},

                'node': {'kind': 'op', 'type': 'Select'},
                'node_data': {'shape': condition_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('input_cond', 'input_cond_data'),
                                   ('input_than', 'input_than_data'),
                                   ('input_else', 'input_else_data'),
                                   ('input_cond_data', 'node'),
                                   ('input_than_data', 'node'),
                                   ('input_else_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')])

        return onnx_net, ref_net

    test_data = [dict(condition_shape=[4, 6], shape_than=[4, 6], else_shape=[4, 6]),
                 dict(condition_shape=[4, 6], shape_than=[4, 6], else_shape=[1, 6]),
                 dict(condition_shape=[15, 3, 5], shape_than=[15, 1, 5], else_shape=[15, 3, 5]),
                 dict(condition_shape=[2, 3, 4, 5], shape_than=[], else_shape=[2, 3, 4, 5]),
                 dict(condition_shape=[2, 3, 4, 5], shape_than=[5], else_shape=[2, 3, 4, 5]),
                 dict(condition_shape=[2, 3, 4, 5], shape_than=[2, 1, 1, 5],
                      else_shape=[2, 3, 4, 5]),
                 dict(condition_shape=[2, 3, 4, 5], shape_than=[2, 3, 4, 5],
                      else_shape=[1, 3, 1, 5]),
                 ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_where(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)
