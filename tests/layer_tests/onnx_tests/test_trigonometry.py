# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestTrigonomery(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.rand(*(inputs_dict[input])).astype(np.float32)
        return inputs_dict

    def create_net(self, shape, op, ir_version):

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        assert op in ['Sin', 'Sinh', 'Asin', 'Cos', 'Cosh', 'Acos', 'Tan', 'Tanh', 'Atan']

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        node_def = onnx.helper.make_node(
            op,
            inputs=['input'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #
        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'node': {'kind': 'op', 'type': op},
                'node_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')])

        return onnx_net, ref_net

    def create_net_const(self, shape, op, precision, ir_version):
        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        assert op in ['Sin', 'Sinh', 'Asin', 'Cos', 'Cosh', 'Acos', 'Tan', 'Tanh', 'Atan']

        concat_axis = 0
        output_shape = shape.copy()
        output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        constant = np.random.rand(*shape).astype(float)

        node_const_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=constant.shape,
                vals=constant.flatten(),
            ),
        )

        node_def = onnx.helper.make_node(
            op,
            inputs=['const'],
            outputs=['res']
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'res'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_def, node_concat_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #
        if op == 'Sin':
            constant = np.sin(constant)
        elif op == 'Sinh':
            constant = np.sinh(constant)
        elif op == 'Asin':
            constant = np.arcsin(constant)
        elif op == 'Cos':
            constant = np.cos(constant)
        elif op == 'Cosh':
            constant = np.cosh(constant)
        elif op == 'Acos':
            constant = np.arccos(constant)
        elif op == 'Tan':
            constant = np.tan(constant)
        elif op == 'Tanh':
            constant = np.tanh(constant)
        elif op == 'Atan':
            constant = np.arctan(constant)
        if precision == 'FP16':
            constant = constant.astype(np.float16)

        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'input_const_data': {'kind': 'data', 'value': constant.flatten()},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': shape, 'kind': 'data'},
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

    test_data_precommit = [dict(shape=[2, 4, 6, 8])]

    test_data = [dict(shape=[10, 12]),
                 dict(shape=[8, 10, 12]),
                 dict(shape=[6, 8, 10, 12]),
                 dict(shape=[4, 6, 8, 10, 12])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_sin(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, op='Sin'), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_sinh(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, op='Sinh'), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_asin(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, op='Asin'), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_cos_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, op='Cos'), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_cos(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, op='Cos'), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_cosh(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, op='Cosh'), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_acos(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, op='Acos'), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_tan(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, op='Tan'), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_tanh(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, op='Tanh'), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_atan(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version, op='Atan'), ie_device,
                   precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_sin_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, ir_version=ir_version, precision=precision, op='Sin'),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_sinh_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, ir_version=ir_version, precision=precision, op='Sinh'),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_asin_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, ir_version=ir_version, precision=precision, op='Asin'),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_cos_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, ir_version=ir_version, precision=precision, op='Cos'),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_cos_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, ir_version=ir_version, precision=precision, op='Cos'),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_cosh_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, ir_version=ir_version, precision=precision, op='Cosh'),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_acos_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, ir_version=ir_version, precision=precision, op='Acos'),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_tan_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, ir_version=ir_version, precision=precision, op='Tan'),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_tanh_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, ir_version=ir_version, precision=precision, op='Tanh'),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_atan_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_net_const(**params, ir_version=ir_version, precision=precision, op='Atan'),
            ie_device, precision, ir_version, temp_dir=temp_dir)
