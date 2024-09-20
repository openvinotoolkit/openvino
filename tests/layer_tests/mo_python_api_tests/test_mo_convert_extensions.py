# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.runtime as ov
import pytest
from common.mo_convert_test_class import CommonMOConvertTest
from common.onnx_layer_test_class import save_to_onnx
from openvino.runtime import PartialShape, Model


class TestONNXExtensions(CommonMOConvertTest):
    def create_onnx_model(self, tmp_dir):
        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        shape = [2, 3, 4]

        input = helper.make_tensor_value_info(
            'input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info(
            'output', TensorProto.FLOAT, shape)

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
        import openvino.runtime.opset14 as ops

        def custom_converter(node: NodeContext):
            input = node.get_input(0)
            relu = ops.relu(input)
            return [relu.output(0)]

        return ConversionExtension("LeakyRelu", custom_converter)

    def create_custom_op_extension_leaky_relu_to_relu():
        # replaces LeakyRelu with Relu
        from openvino.frontend import OpExtension

        return OpExtension("Relu", "LeakyRelu")

    def create_custom_extension_elu_to_sigmoid():
        # replaces Elu with Sigmoid
        from openvino.frontend import ConversionExtension
        from openvino.frontend import NodeContext
        import openvino.runtime.opset14 as ops

        def custom_converter(node: NodeContext):
            input = node.get_input(0)
            sigm = ops.sigmoid(input)
            return [sigm.output(0)]

        return ConversionExtension("Elu", custom_converter)

    def create_ref_graph1():
        shape = PartialShape([2, 3, 4])
        param = ov.opset14.parameter(shape, dtype=np.float32)
        param.get_output_tensor(0).set_names({"input"})
        relu = ov.opset14.relu(param)
        relu.get_output_tensor(0).set_names({"LeakyRelu_data"})
        elu = ov.opset14.elu(relu, alpha=0.1)
        elu.get_output_tensor(0).set_names({"output"})

        return Model([elu], [param], "test")

    def create_ref_graph2():
        shape = PartialShape([2, 3, 4])
        param = ov.opset14.parameter(shape, dtype=np.float32)
        param.get_output_tensor(0).set_names({"input"})
        relu = ov.opset14.relu(param)
        relu.get_output_tensor(0).set_names({"LeakyRelu_data"})
        sigmoid = ov.opset14.sigmoid(relu)
        sigmoid.get_output_tensor(0).set_names({"output"})

        return Model([sigmoid], [param], "test")

    test_data = [
        {'params_test': {'extensions': create_custom_extension_leaky_relu_to_relu()},
         'ref_graph': create_ref_graph1()},
        {'params_test': {'extensions': create_custom_op_extension_leaky_relu_to_relu()},
         'ref_graph': create_ref_graph1()},
        {'params_test': {'extensions': [create_custom_extension_leaky_relu_to_relu(),
                                        create_custom_extension_elu_to_sigmoid()]},
         'ref_graph': create_ref_graph2()}
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_onnx_mo_convert_extensions(self, params, ie_device, precision, ir_version,
                                        temp_dir, use_legacy_frontend):
        onnx_net_path = self.create_onnx_model(temp_dir)

        test_params = params['params_test']
        test_params.update({'input_model': onnx_net_path})
        test_params.update({'use_convert_model_from_mo': True})
        self._test_by_ref_graph(temp_dir, test_params, params['ref_graph'])


class TestPyTorchExtensions(CommonMOConvertTest):
    def create_model(self, tmp_dir):
        import torch

        class CosModel(torch.nn.Module):
            def __init__(self):
                super(CosModel, self).__init__()

            def forward(self, x):
                return torch.cos(x.to(torch.float32))

        return CosModel()

    def create_custom_extension_cos_to_sin():
        from openvino.frontend import ConversionExtension
        from openvino.frontend import NodeContext
        import openvino.runtime.opset14 as ops

        def custom_converter(node: NodeContext):
            input = node.get_input(0)
            sin = ops.sin(input)
            return sin.outputs()

        return ConversionExtension("aten::cos", custom_converter)

    def create_custom_op_extension_cos_to_sin():
        from openvino.frontend import OpExtension

        return OpExtension("Sin", "aten::cos")

    def create_ref_graph():
        shape = PartialShape.dynamic()
        param = ov.opset14.parameter(shape, dtype=ov.Type.dynamic)
        param.get_output_tensor(0).set_names({"x"})
        convert = ov.opset14.convert(param, ov.Type.f32)
        convert.get_output_tensor(0).set_names({"5"})
        sin = ov.opset14.sin(convert)

        return Model([sin], [param], "test")

    test_data = [
        {'params_test': {'extension': create_custom_extension_cos_to_sin()},
         'ref_graph': create_ref_graph()},
        {'params_test': {'extension': create_custom_op_extension_cos_to_sin()},
         'ref_graph': create_ref_graph()},
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_pt_mo_convert_extensions(self, params, ie_device, precision, ir_version,
                                      temp_dir, use_legacy_frontend):
        model = self.create_model(temp_dir)

        test_params = params['params_test']
        test_params.update({'input_model': model})
        self._test_by_ref_graph(temp_dir, test_params, params['ref_graph'])


class TestTfExtensions(CommonMOConvertTest):
    def create_keras_model(self, temp_dir):
        import tensorflow as tf

        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

        input_name = "Input1"
        input_shape = [None, 1, 2, 3]

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, input_shape, input_name)
            tf.raw_ops.Cos(x=x, name='res')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net

    def create_custom_extension_cos_to_sin():
        from openvino.frontend import ConversionExtension
        from openvino.frontend import NodeContext
        import openvino.runtime.opset14 as ops

        def custom_converter(node: NodeContext):
            input = node.get_input(0)
            sin = ops.sin(input)
            return sin.outputs()

        return ConversionExtension("Cos", custom_converter)

    def create_custom_op_extension_cos_to_sin():
        from openvino.frontend import OpExtension

        return OpExtension("Sin", "Cos")

    def create_ref_graph():
        shape = PartialShape([-1, 1, 2, 3])
        param = ov.opset14.parameter(shape, dtype=np.float32)
        param.get_output_tensor(0).set_names({"Input1:0"})
        y = ov.opset14.sin(param)

        parameter_list = [param]

        return Model([y], parameter_list, "test")

    test_data = [
        {'params_test': {'extension': create_custom_extension_cos_to_sin()},
         'ref_graph': create_ref_graph()},
        {'params_test': {'extension': create_custom_op_extension_cos_to_sin()},
         'ref_graph': create_ref_graph()},
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_tf_mo_convert_extensions(self, params, ie_device, precision, ir_version,
                                      temp_dir, use_legacy_frontend):
        model = self.create_keras_model(temp_dir)

        test_params = params['params_test']
        test_params.update({'input_model': model})
        self._test_by_ref_graph(temp_dir, test_params, params['ref_graph'])
