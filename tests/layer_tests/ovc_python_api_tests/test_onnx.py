# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import tempfile
import unittest

import numpy as np
import openvino.runtime as ov
import pytest
from openvino.runtime import Model
from openvino.test_utils import compare_functions

from common.mo_convert_test_class import CommonMOConvertTest


def make_graph_proto_model():
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

    return onnx_net


def create_ref_model(shape):
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    slope_const = ov.opset8.constant([0.1], dtype=np.float16)
    decompress_slope = ov.opset8.convert(slope_const, np.float32)
    prelu = ov.opset8.prelu(param1, slope=decompress_slope)
    relu = ov.opset8.elu(prelu, alpha=np.float32(0.1))
    parameter_list = [param1]
    return Model([relu], parameter_list, "test")


def create_ref_model_fp32(shape):
    param1 = ov.opset8.parameter(shape, dtype=np.float32)
    slope_const = ov.opset8.constant([0.1], dtype=np.float32)
    prelu = ov.opset8.prelu(param1, slope=slope_const)
    relu = ov.opset8.elu(prelu, alpha=np.float32(0.1))
    parameter_list = [param1]
    return Model([relu], parameter_list, "test")


def create_bytes_io():
    import onnx
    onnx_model = make_graph_proto_model()

    file_like_object = io.BytesIO()
    onnx.save(onnx_model, file_like_object)

    ref_model = create_ref_model([2,3,4])
    return file_like_object, ref_model, {}


class TestMoConvertONNX(CommonMOConvertTest):
    test_data = ['create_bytes_io']
    @pytest.mark.parametrize("create_model", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_convert_onnx(self, create_model, ie_device, precision, ir_version,
                                             temp_dir):
        fw_model, graph_ref, mo_params = eval(create_model)()

        test_params = {'input_model': fw_model}
        if mo_params is not None:
            test_params.update(mo_params)
        self._test_by_ref_graph(temp_dir, test_params, graph_ref, compare_tensor_names=False)


class TestUnicodePathsONNX(unittest.TestCase):
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_unicode_paths(self):
        import os
        import onnx
        test_directory = os.path.dirname(os.path.realpath(__file__))
        with tempfile.TemporaryDirectory(dir=test_directory, prefix=r"晚安_путь_к_файлу") as temp_dir:
            model = make_graph_proto_model()
            model_path = temp_dir + os.sep +'test_model.onnx'
            model_ref = create_ref_model_fp32([2, 3, 4])

            try:
                onnx.save(model, model_path)
            except:
                return

            assert os.path.exists(model_path), "Could not create a directory with unicode path."

            from openvino import convert_model, save_model, Core
            res_model = convert_model(model_path)
            flag, msg = compare_functions(res_model, model_ref, False)
            assert flag, msg

            save_model(res_model, model_path + ".xml")
            res_model_after_saving = Core().read_model(model_path + ".xml")
            flag, msg = compare_functions(res_model_after_saving, create_ref_model([2, 3, 4]), False)
            assert flag, msg

            from openvino.frontend import FrontEndManager
            fm = FrontEndManager()
            fe = fm.load_by_framework("onnx")

            assert fe.supported(model_path)

            del res_model_after_saving
