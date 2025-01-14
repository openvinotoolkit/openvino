# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestAnd(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(0, 2, inputs_dict[input]).astype(bool)
        return inputs_dict

    def create_net(self, shape1, shape2, ir_version):
        """
            ONNX net                                  IR net

            Input->And with 2nd input->Output   =>    Input->LogicalAnd
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        input1 = helper.make_tensor_value_info('input1', TensorProto.BOOL, shape1)
        input2 = helper.make_tensor_value_info('input2', TensorProto.BOOL, shape2)
        output = helper.make_tensor_value_info('output', TensorProto.BOOL, shape1)

        node_def = helper.make_node(
            'And',
            inputs=['input1', 'input2'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [input1, input2],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #   Create reference IR net

        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input1': {'kind': 'op', 'type': 'Parameter'},
                'input1_data': {'shape': shape1, 'kind': 'data'},
                'input2': {'kind': 'op', 'type': 'Parameter'},
                'input2_data': {'shape': shape2, 'kind': 'data'},
                'node': {'kind': 'op', 'type': 'LogicalAnd'},
                'node_data': {'shape': shape1, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('input1', 'input1_data'),
                                   ('input2', 'input2_data'),
                                   ('input1_data', 'node'),
                                   ('input2_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')])

        return onnx_net, ref_net

    def create_net_one_const(self, shape1, shape2, ir_version):
        """
            ONNX net                              IR net

            Input->And with const->Output   =>    Input->LogicalAnd
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.BOOL, shape1)
        output = helper.make_tensor_value_info('output', TensorProto.BOOL, shape1)

        const = np.random.randint(0, 2, shape2).astype(bool)

        node_const_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.BOOL,
                dims=const.shape,
                vals=const.flatten(),
            ),
        )

        node_def = helper.make_node(
            'And',
            inputs=['input', 'const'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #   Create reference IR net

        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape1, 'kind': 'data'},
                'input_const_data': {'kind': 'data', 'value': const.flatten()},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': const.shape, 'kind': 'data'},
                'node': {'kind': 'op', 'type': 'LogicalAnd'},
                'node_data': {'shape': shape1, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_const_data', 'const'),
                                   ('const', 'const_data'),
                                   ('input_data', 'node'),
                                   ('const_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')])

        return onnx_net, ref_net

    def create_net_const(self, shape1, shape2, ir_version):
        """
            ONNX net                                          IR net

            Input->Concat with const and const->Output   =>   Input->Concat
        """

        #
        #   Create ONNX model
        #

        from onnx import helper
        from onnx import TensorProto

        concat_axis = 0
        output_shape = list(shape1)
        output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.BOOL, shape1)
        output = helper.make_tensor_value_info('output', TensorProto.BOOL, output_shape)

        const1 = np.random.randint(0, 2, shape1).astype(bool)
        const2 = np.random.randint(0, 2, shape2).astype(bool)

        node_const1_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.BOOL,
                dims=const1.shape,
                vals=const1.flatten(),
            ),
        )

        node_const2_def = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const2'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.BOOL,
                dims=const2.shape,
                vals=const2.flatten(),
            ),
        )

        node_def = helper.make_node(
            'And',
            inputs=['const1', 'const2'],
            outputs=['node_out']
        )

        node_concat_def = helper.make_node(
            'Concat',
            inputs=['input', 'node_out'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const1_def, node_const2_def, node_def, node_concat_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #   Create reference IR net
        constant_calculated = np.logical_and(const1, const2)

        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': const1.shape, 'kind': 'data'},
                'input_const_data': {'kind': 'data', 'value': constant_calculated.flatten()},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': const1.shape, 'kind': 'data'},
                'concat': {'kind': 'op', 'type': 'Concat', 'axis': concat_axis},
                'concat_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_const_data', 'const'),
                                   ('const', 'const_data'),
                                   ('input_data', 'concat'),
                                   ('const_data', 'concat'),
                                   ('concat', 'concat_data'),
                                   ('concat_data', 'result')])

        return onnx_net, ref_net

    test_data = [dict(shape1=[4, 6], shape2=[4, 6]),
                 dict(shape1=[4, 6, 8], shape2=[4, 6, 8]),
                 dict(shape1=[4, 6, 8, 10], shape2=[4, 6, 8, 10]),
                 dict(shape1=[4, 6, 8, 10, 12], shape2=[4, 6, 8, 10, 12])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_and(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_and_one_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_one_const(**params, ir_version=ir_version), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_and_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)
