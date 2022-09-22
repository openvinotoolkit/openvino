# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

from generator import generator, generate
from openvino.runtime import serialize

from openvino.tools.mo import InputCutInfo, LayoutMap
from openvino.tools.mo.utils.ir_engine.ir_engine import IREngine
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from unit_tests.utils.graph import build_graph
from utils import create_onnx_model, save_to_onnx


@generator
class ConvertImportMOTest(UnitTestWithMockedTelemetry):
    # Checks convert import from openvino.tools.mo
    test_directory = os.path.dirname(os.path.realpath(__file__))

    @generate(*[
        ({}),
        ({'input': InputCutInfo(name='LeakyRelu_out', shape=None, type=None, value=None)}),
        ({'layout': {'input': LayoutMap(source_layout='NCHW', target_layout='NHWC')}}),
    ])
    def test_import(self, params):
        from openvino.tools.mo import convert

        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:
            model = create_onnx_model()
            model_path = save_to_onnx(model, tmpdir)
            out_xml = os.path.join(tmpdir, "model.xml")

            ov_model = convert(input_model=model_path, **params)
            serialize(ov_model, out_xml.encode('utf-8'), out_xml.replace('.xml', '.bin').encode('utf-8'))
            assert os.path.exists(out_xml)

    def test_unnamed_input_model(self):
        def create_onnx_model():
            #
            #   Create ONNX model
            #

            import onnx
            from onnx import helper
            from onnx import TensorProto

            shape = [1, 2, 3]

            input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
            output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

            node_def = onnx.helper.make_node(
                'Relu',
                inputs=['input'],
                outputs=['Relu_out'],
            )
            node_def2 = onnx.helper.make_node(
                'Sigmoid',
                inputs=['Relu_out'],
                outputs=['output'],
            )

            # Create the graph (GraphProto)
            graph_def = helper.make_graph(
                [node_def, node_def2],
                'test_model',
                [input],
                [output],
            )

            # Create the model (ModelProto)
            onnx_net = helper.make_model(graph_def, producer_name='test_model')
            return onnx_net

        nodes_attributes = {
            'input': {'kind': 'op', 'type': 'Parameter'},
            'input_data': {'shape': [1, 2, 3], 'kind': 'data'},
            'relu': {'kind': 'op', 'type': 'ReLU'},
            'relu_data': {'shape': [1, 2, 3], 'kind': 'data'},
            'sigmoid': {'kind': 'op', 'type': 'Sigmoid'},
            'sigmoid_data': {'shape': [1, 2, 3], 'kind': 'data'},
            'result': {'kind': 'op', 'type': 'Result'}
        }

        ref_graph = build_graph(nodes_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'relu'),
                                 ('relu', 'relu_data'),
                                 ('relu_data', 'sigmoid'),
                                 ('sigmoid', 'sigmoid_data'),
                                 ('sigmoid_data', 'result'),
                                 ])

        from openvino.tools.mo import convert
        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:

            model = create_onnx_model()
            model_path = save_to_onnx(model, tmpdir)
            out_xml = os.path.join(tmpdir, "model.xml")

            ov_model = convert(model_path)
            serialize(ov_model, out_xml.encode('utf-8'), out_xml.replace('.xml', '.bin').encode('utf-8'))

            ir = IREngine(out_xml, out_xml.replace('.xml', '.bin'))
            flag, resp = ir.compare(ref_graph)
            assert flag, '\n'.join(resp)
