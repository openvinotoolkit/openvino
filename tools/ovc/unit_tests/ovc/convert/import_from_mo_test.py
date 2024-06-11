# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from openvino.runtime import serialize
from pathlib import Path
from unit_tests.ovc.convert.utils import create_onnx_model, save_to_onnx
from unit_tests.ovc.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry


class ConvertImportMOTest(UnitTestWithMockedTelemetry):
    test_directory = os.path.dirname(os.path.realpath(__file__))

    @staticmethod
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

    # Checks convert import from openvino.tools.mo
    def test_import(self):
        from openvino.tools.ovc import convert_model

        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:
            model = create_onnx_model()
            model_path = save_to_onnx(model, tmpdir)
            out_xml = os.path.join(tmpdir, "model.xml")

            ov_model = convert_model(input_model=model_path)
            serialize(ov_model, out_xml.encode('utf-8'), out_xml.replace('.xml', '.bin').encode('utf-8'))
            assert os.path.exists(out_xml)

    def test_input_model_path(self):
        from openvino.tools.ovc import convert_model

        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:
            model = self.create_onnx_model()
            model_path = save_to_onnx(model, tmpdir)
            out_xml = os.path.join(tmpdir, "model.xml")

            ov_model = convert_model(Path(model_path))
            serialize(ov_model, out_xml.encode('utf-8'), out_xml.replace('.xml', '.bin').encode('utf-8'))

            # TODO: check that model is correct

    def test_unnamed_input_model(self):
        from openvino.tools.ovc import convert_model
        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:
            model = self.create_onnx_model()
            model_path = save_to_onnx(model, tmpdir)
            out_xml = os.path.join(tmpdir, "model.xml")

            ov_model = convert_model(model_path)
            # serialize(ov_model, out_xml.encode('utf-8'), out_xml.replace('.xml', '.bin').encode('utf-8'))

            # TODO: check that model is correct
