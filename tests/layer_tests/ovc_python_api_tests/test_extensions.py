# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from common.mo_convert_test_class import CommonMOConvertTest
from common.onnx_layer_test_class import save_to_onnx

import openvino.runtime as ov
from openvino.runtime import PartialShape, Model


class TestExtensions(CommonMOConvertTest):
    def create_onnx_model(self, tmp_dir):
        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        shape = [2, 3, 4]

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        node_def = onnx.helper.make_node(
            'LeakyRelu',
            inputs=['input'],
            outputs=['LeakyRelu_data'],
            alpha=0.1
        )
        node_def2 = onnx.helper.make_node(
            'Elu',
            inputs=['LeakyRelu_data'],
            outputs=['output'],
            alpha=0.1
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

        # save model to .onnx and return path to the model
        return save_to_onnx(onnx_net, tmp_dir)

    def create_custom_extension_leaky_relu_to_relu():
        # replaces LeakyRelu with Relu
        from openvino.frontend import ConversionExtension
        from openvino.frontend import NodeContext
        import openvino.runtime.opset8 as ops

        def custom_converter(node: NodeContext):
            input = node.get_input(0)
            relu = ops.relu(input)
            return [relu.output(0)]

        return ConversionExtension("LeakyRelu", custom_converter)

    def create_custom_extension_elu_to_sigmoid():
        # replaces Elu with Sigmoid
        from openvino.frontend import ConversionExtension
        from openvino.frontend import NodeContext
        import openvino.runtime.opset8 as ops

        def custom_converter(node: NodeContext):
            input = node.get_input(0)
            sigm = ops.sigmoid(input)
            return [sigm.output(0)]

        return ConversionExtension("Elu", custom_converter)

    def create_ref_graph1():
        shape = PartialShape([2, 3, 4])
        param = ov.opset8.parameter(shape, dtype=np.float32)
        param.get_output_tensor(0).set_names({"input"})
        relu = ov.opset8.relu(param)
        relu.get_output_tensor(0).set_names({"LeakyRelu_data"})
        elu = ov.opset8.elu(relu, alpha=0.1)
        elu.get_output_tensor(0).set_names({"output"})

        return Model([elu], [param], "test")

    def create_ref_graph2():
        shape = PartialShape([2, 3, 4])
        param = ov.opset8.parameter(shape, dtype=np.float32)
        param.get_output_tensor(0).set_names({"input"})
        relu = ov.opset8.relu(param)
        relu.get_output_tensor(0).set_names({"LeakyRelu_data"})
        sigmoid = ov.opset8.sigmoid(relu)
        sigmoid.get_output_tensor(0).set_names({"output"})

        return Model([sigmoid], [param], "test")

    test_data = [
        {'params_test': {'extension': create_custom_extension_leaky_relu_to_relu()},
         'ref_graph': create_ref_graph1()},
        {'params_test': {'extension': [create_custom_extension_leaky_relu_to_relu(),
                                        create_custom_extension_elu_to_sigmoid()]},
         'ref_graph': create_ref_graph2()}
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_convert_extensions(self, params, ie_device, precision, ir_version,
                                   temp_dir, use_legacy_frontend):
        onnx_net_path = self.create_onnx_model(temp_dir)

        test_params = params['params_test']
        test_params.update({'input_model': onnx_net_path})
        self._test_by_ref_graph(temp_dir, test_params, params['ref_graph'])
