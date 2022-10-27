# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

from generator import generator
from openvino.runtime import get_version as get_rt_version
from openvino.runtime import serialize

from openvino.tools.mo import convert
from openvino.tools.mo.utils import import_extensions
from openvino.tools.mo.utils.version import get_version
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from utils import save_to_onnx

from openvino.tools.mo.utils.ir_reader.restore_graph import restore_graph_from_ir, save_restored_graph


@generator
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
                'legacy_path': "False",
                'conversion_parameters': {
                    'caffe_parser_path': Path("DIR"),
                    'compress_fp16': "False",
                    'data_type': "float",
                    'disable_nhwc_to_nchw': "False",
                    'disable_omitting_optional': "False",
                    'disable_resnet_optimization': "False",
                    'disable_weights_compression': "False",
                    'enable_concat_optimization': "False",
                    'enable_flattening_nested_params': "False",
                    'enable_ssd_gluoncv': "False",
                    'extensions': Path("['" + import_extensions.default_path() + "']"),
                    'framework': "onnx",
                    'freeze_placeholder_with_value': "{}",
                    'input_model': Path.joinpath(Path("DIR"), Path("model.onnx")),
                    'input_model_is_text': "False",
                    'inputs_list': "[]",
                    'k': Path.joinpath(Path("DIR"), Path("CustomLayersMapping.xml")),
                    'layout': "()",
                    'layout_values': "{}",
                    'legacy_mxnet_model': "False",
                    'log_level': "ERROR",
                    'mean_scale_values': "{}",
                    'mean_values': "()",
                    'model_name': "model",
                    'output_dir': Path("DIR"),
                    'placeholder_data_types': "{}",
                    'progress': "False",
                    'remove_memory': "False",
                    'remove_output_softmax': "False",
                    'reverse_input_channels': "False",
                    'save_params_from_nd': "False",
                    'scale_values': "()",
                    'silent': "True",
                    'source_layout': "()",
                    'static_shape': "False",
                    'stream_output': "False",
                    'target_layout': "()",
                    'transform': "",
                    'unset': "['input_shape', 'scale', 'input', 'output', 'disable_fusing', 'finegrain_fusing', "
                             "'batch', 'transformations_config', 'input_checkpoint', 'input_meta_graph', "
                             "'saved_model_dir', 'saved_model_tags', 'tensorflow_custom_operations_config_update', "
                             "'tensorflow_use_custom_operations_config', "
                             "'tensorflow_object_detection_api_pipeline_config', 'tensorboard_logdir', "
                             "'tensorflow_custom_layer_libraries', 'input_proto', 'mean_file', 'mean_file_offsets', "
                             "'input_symbol', 'nd_prefix_name', 'pretrained_model_name', 'counts', "
                             "'placeholder_shapes']",
                    'use_legacy_frontend': "False",
                    'use_new_frontend': "False",

                }

            }

        def check_meta_data(ov_model):
            ref_meta = ref_meta_data()
            for key, value in ref_meta.items():
                if key == 'conversion_parameters':
                    for param_name, param_value in value.items():
                        val = ov_model.get_rt_info([key, param_name])
                        if param_name in ['extensions', 'caffe_parser_path', 'input_model', 'k', 'output_dir']:
                            val = Path(val)
                        assert val == param_value, \
                            "Runtime info attribute with name {} does not match. Expected: {}, " \
                            "got {}".format(param_name, param_value, val)
                    continue
                assert str(ov_model.get_rt_info(key)) == value, \
                    "Runtime info attribute with name {} does not match. Expected: {}, " \
                    "got {}".format(key, value, ov_model.get_rt_info(key))

        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:

            model = create_onnx_model()
            model_path = save_to_onnx(model, tmpdir)
            out_xml = os.path.join(tmpdir, "model.xml")

            ov_model = convert(model_path)
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
