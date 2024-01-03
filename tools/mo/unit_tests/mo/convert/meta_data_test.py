# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

from openvino.runtime import get_version as get_rt_version
from openvino.runtime import serialize

from openvino.tools.mo import convert_model
from openvino.tools.mo.utils import import_extensions
from openvino.tools.mo.utils.version import get_version
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from utils import save_to_onnx

from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph


class MetaDataTest(UnitTestWithMockedTelemetry):
    test_directory = os.path.dirname(os.path.realpath(__file__))

    def test_meta_data(self):
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

        def ref_meta_data():
            return {
                'MO_version': get_version(),
                'Runtime_version': get_rt_version(),
                'legacy_frontend': "False",
                'conversion_parameters': {
                    'input_model': Path.joinpath(Path("DIR"), Path("model.onnx")),
                }

            }

        def check_meta_data(ov_model):
            ref_meta = ref_meta_data()
            for key, value in ref_meta.items():
                if key == 'conversion_parameters':
                    for param_name, param_value in value.items():
                        val = ov_model.get_rt_info([key, param_name]).astype(str)
                        if param_name in ['extensions', 'caffe_parser_path', 'input_model', 'k', 'output_dir']:
                            val = Path(val)
                        assert val == param_value, \
                            "Runtime info attribute with name {} does not match. Expected: {}, " \
                            "got {}".format(param_name, param_value, val)
                    continue
                assert ov_model.get_rt_info(key).astype(str) == value, \
                    "Runtime info attribute with name {} does not match. Expected: {}, " \
                    "got {}".format(key, value, ov_model.get_rt_info(key).astype(str))

        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:

            model = create_onnx_model()
            model_path = save_to_onnx(model, tmpdir)
            out_xml = os.path.join(tmpdir, "model.xml")

            ov_model = convert_model(model_path)
            check_meta_data(ov_model)

            serialize(ov_model, out_xml.encode('utf-8'), out_xml.replace('.xml', '.bin').encode('utf-8'))

            from openvino.runtime import Core
            core = Core()
            serialized_model = core.read_model(out_xml)
            check_meta_data(serialized_model)

            restored_graph, meta_data = restore_graph_from_ir(out_xml, out_xml.replace('.xml', '.bin'))
            save_restored_graph(restored_graph, tmpdir, meta_data, "mo_ir_reader_test_model")

            mo_ir_reader_test_model = core.read_model(os.path.join(tmpdir, "mo_ir_reader_test_model.xml"))
            check_meta_data(mo_ir_reader_test_model)
