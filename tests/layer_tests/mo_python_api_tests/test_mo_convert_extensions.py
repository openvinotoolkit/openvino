# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.runtime as ov
import os
import pytest
import subprocess
import tempfile
import tensorflow as tf
import unittest
from openvino.runtime import PartialShape, Model
from openvino.test_utils import compare_functions
from unit_tests.utils.graph import build_graph

from common.mo_convert_test_class import CommonMOConvertTest
from common.onnx_layer_test_class import save_to_onnx
from common.utils.common_utils import generate_ir


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
        {'params_test': {'extensions': create_custom_extension_leaky_relu_to_relu()},
         'ref_graph': create_ref_graph1()},
        {'params_test': {'extensions': [create_custom_extension_leaky_relu_to_relu(),
                                        create_custom_extension_elu_to_sigmoid()]},
         'ref_graph': create_ref_graph2()}
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_convert_extensions(self, params, ie_device, precision, ir_version,
                                   temp_dir, use_new_frontend, use_old_api):
        onnx_net_path = self.create_onnx_model(temp_dir)

        test_params = params['params_test']
        test_params.update({'input_model': onnx_net_path})
        self._test_by_ref_graph(temp_dir, test_params, params['ref_graph'])


def create_tf_model():
    import tensorflow as tf

    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        inp1 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
        inp2 = tf.compat.v1.placeholder(tf.float32, [1, 2, 3], 'Input')
        relu = tf.nn.relu(inp1 + inp2, name='Relu')

        output = tf.nn.sigmoid(relu, name='Sigmoid')

        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph_def
    return tf_net


def create_ref_model_1():
    shape = [1, 2, 3]
    shape = PartialShape(shape)
    param1 = ov.opset10.parameter(shape)
    param2 = ov.opset10.parameter(shape)
    add = ov.opset10.add(param1, param2)
    relu = ov.opset10.relu(add)
    sin = ov.opset10.sin(relu)
    sigm = ov.opset10.sigmoid(sin)
    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return ref_model


def create_ref_model_2():
    shape = [1, 2, 3]
    shape = PartialShape(shape)
    param1 = ov.opset10.parameter(shape)
    param2 = ov.opset10.parameter(shape)
    add = ov.opset10.add(param1, param2)
    relu = ov.opset10.relu(add)
    sin = ov.opset10.sin(relu)
    sigm = ov.opset10.sigmoid(sin)
    tanh = ov.opset10.tanh(sigm)
    parameter_list = [param1, param2]
    ref_model = Model([tanh], parameter_list, "test")
    return ref_model


class LegacyExtTest(unittest.TestCase):
    test_directory = os.path.dirname(os.path.realpath(__file__))
    def test_legacy_extensions(self):
        from openvino.tools.mo import convert_model
        with tempfile.TemporaryDirectory(dir=self.test_directory) as tmpdir:
            ext_path1 = os.path.join(os.path.dirname(__file__), "test_legacy_exts/test_exts_dir1")
            ext_path2 = os.path.join(os.path.dirname(__file__), "test_legacy_exts/test_exts_dir2")
            model = create_tf_model()
            out_xml = os.path.join(tmpdir, "model.xml")

            # tests for convert_model()
            ov_model = convert_model(model, extensions=ext_path1)
            flag, msg = compare_functions(ov_model, create_ref_model_1(), False)
            assert flag, msg

            ov_model = convert_model(model, extensions=[ext_path1, ext_path2])
            flag, msg = compare_functions(ov_model, create_ref_model_2(), False)
            assert flag, msg

            ov_model = convert_model(model, extensions=','.join([ext_path1, ext_path2]))
            flag, msg = compare_functions(ov_model, create_ref_model_2(), False)
            assert flag, msg

            tf.io.write_graph(model, tmpdir, 'model.pb', False)
            inp_model = os.path.join(tmpdir, 'model.pb')
            from openvino.runtime import Core
            core = Core()

            # tests for MO cli tool
            exit_code, stderr = generate_ir(coverage=False, **{"input_model": inp_model,
                                                               "extensions": ext_path1,
                                                               "output_dir": tmpdir})
            assert not exit_code

            ov_model = core.read_model(os.path.join(tmpdir, "model.xml"))
            flag, msg = compare_functions(ov_model, create_ref_model_1(), False)
            assert flag, msg

            exit_code, stderr = generate_ir(coverage=False, **{"input_model": inp_model,
                                                               "extensions": ','.join([ext_path1, ext_path2]),
                                                               "output_dir": tmpdir})
            assert not exit_code

            ov_model = core.read_model(os.path.join(tmpdir, "model.xml"))
            flag, msg = compare_functions(ov_model, create_ref_model_2(), False)
            assert flag, msg


