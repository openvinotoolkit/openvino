# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np
import openvino.runtime as ov
import pytest
import torch
from openvino.runtime import PartialShape, Dimension, Model, Type, Core, save_model
from openvino.test_utils import compare_functions

from openvino.tools.ovc import convert_model


def create_pytorch_layer_norm():
    class aten_layer_norm(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.layer_norm(x, normalized_shape=[3])

    shape = PartialShape(PartialShape([-1, -1]))
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.float32)
    const1 = ov.opset8.constant([-1], dtype=np.int32)
    mvn1 = ov.opset8.mvn(param1, const1, True, 1e-5, "inside_sqrt")
    ref_model = Model([mvn1], [param1], "test")

    test_params = {'example_input': 300 + np.random.randn(2, 3).astype(np.float32)}
    return aten_layer_norm(), ref_model, test_params


def create_pytorch_normalize():
    class aten_normalize(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.normalize(x)

    test_params = {'example_input': 300 + np.random.randn(2, 3).astype(np.float32)}
    return aten_normalize(), None, test_params


def create_pytorch_precision_sensitive_with_div():
    class precision_sensitive_with_div(torch.nn.Module):
        def forward(self, x):
            eps = 1.0e-8
            return 2.0 / (torch.sqrt(torch.sum(torch.pow(x + 2, 2.0), 1)) + eps)
    test_params = {'example_input': 300 + np.random.randn(2, 3).astype(np.float32)}
    return precision_sensitive_with_div(), None, test_params


def create_pytorch_precision_sensitive_for_exp_reduce(tmp_dir):
    class precision_sensitive_for_exp_reduce(torch.nn.Module):
        def forward(self, x):
            return torch.sum(torch.exp(x + 10), 1)

    test_params = {'example_input': 300 + np.random.randn(2, 3).astype(np.float32)}
    return precision_sensitive_for_exp_reduce(), None, test_params


def create_pytorch_precision_sensitive_div_as_pow(tmp_dir):
    class precision_sensitive_div_as_pow(torch.nn.Module):
        def forward(self, x):
            eps = 1.0e-8
            return 2.0 * (torch.sqrt(torch.sum(torch.pow(x + 2, 2.0), 1)) + eps)**(-1)

    test_params = {'example_input': 300 + np.random.randn(2, 3).astype(np.float32)}
    return precision_sensitive_div_as_pow(), None, test_params


def create_pytorch_precision_sensitive_two_inp_1(tmp_dir):
    class precision_sensitive_two_inp_1(torch.nn.Module):
        def forward(self, x, y):
            eps = 1.0e-8
            return x / (torch.sqrt(torch.sum(torch.pow(y + 2, 2.0), 2)) + eps)
    test_params = {'example_input': (10000 + np.ones((2, 10), dtype=np.float32),
                                     300 + np.ones((2, 10, 3), dtype=np.float32))}
    return precision_sensitive_two_inp_1(), None, test_params


def create_pytorch_precision_sensitive_two_inp_2(tmp_dir):
    class precision_sensitive_two_inp_2(torch.nn.Module):
        def forward(self, x, y):
            eps = 1.0e-8
            return x * (torch.sqrt(torch.sum(torch.pow(y + 2, 2.0), 2)) + eps)**(-1)
    test_params = {'example_input': (10000 + np.ones((2, 10), dtype=np.float32),
                                     300 + np.ones((2, 10, 3), dtype=np.float32))}
    return precision_sensitive_two_inp_2(), None, test_params


def create_pytorch_precision_sensitive_with_matmul(tmp_dir):
    class precision_sensitive_with_matmul(torch.nn.Module):
        def forward(self, x, y):
            eps = 1.0e-8
            interm_res = x / (torch.sqrt(torch.sum(torch.pow(y + 2, 2.0), 2)) + eps)
            print(f"interm_res shpae: {interm_res.shape}")
            print(interm_res)
            weights = 1024.0 + torch.zeros(10, 2)
            return torch.mm(interm_res, weights)
    test_params = {'example_input': (10000 + np.ones((2, 10), dtype=np.float32),
                                     300 + np.ones((2, 10, 3), dtype=np.float32))}
    return precision_sensitive_with_matmul(), None, test_params


def create_pytorch_not_precision_sensitive(tmp_dir):
    class not_precision_sensitive(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x, 1)

    test_params = 10000.0 + np.zeros((2, 20), dtype=np.float32),  # 10 000 * 20 = 200 000 > 65504 (fp16_max)
    return not_precision_sensitive(), None, test_params


def make_pt_model_two_inputs():
    from torch import nn

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x, y):
            mm_res = torch.matmul(x, y)
            logits = self.linear_relu_stack(mm_res)
            return logits

    test_params = {'example_input': (750 + np.ones((2, 100), dtype=np.float32),
                                     np.ones((100, 4), dtype=np.float32))}
    return NeuralNetwork(), None, test_params

def make_pt_model_one_input():
    from torch import nn

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.mm_const_val = torch.ones((100, 4), dtype=torch.float32)
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            mm_res = torch.matmul(x, self.mm_const_val)
            logits = self.linear_relu_stack(mm_res)
            return logits

    test_params = {'example_input': (750 + np.ones((2, 100), dtype=np.float32))}
    return NeuralNetwork(), None, test_params


class TestPrecisionSensitive():
    test_data = [
        # make_pt_model_two_inputs,
        make_pt_model_one_input,
    ]

    @pytest.mark.parametrize("create_model", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_precision_sensitive(self, create_model, ie_device, precision, ir_version, temp_dir, use_new_frontend, use_old_api):
        import numpy.testing as npt
        from pathlib import Path

        fw_model, ref_model, mo_params = create_model()

        test_params = {'input_model': fw_model}
        if mo_params is not None:
            test_params.update(mo_params)

        model = convert_model(**test_params)
        model_name = 'model_test.xml'
        compress_to_fp16 = True if precision == 'FP16' else False

        # save_model(model, str(Path(temp_dir, model_name)), compress_to_fp16)
        from openvino._offline_transformations import compress_model_transformation  # pylint: disable=import-error,no-name-in-module
        # compress_model_transformation(model)

        from openvino.tools.ovc.improve_fp16_accuracy import improve_fp16_accuracy
        new_ov_model = improve_fp16_accuracy(model, test_params['example_input'])


        core = Core()
        ir_test = core.read_model(Path(temp_dir, model_name))
        if ref_model is not None:
            flag, msg = compare_functions(ir_test, ref_model, compare_tensor_names=False)
            assert flag, msg

        example_inputs = test_params['example_input']
        torch_inp_tensors = []
        if isinstance(example_inputs, tuple):
            for input_arr in example_inputs:
                torch_inp_tensors.append(torch.tensor(input_arr))
        else:
            torch_inp_tensors.append(torch.tensor(example_inputs))

        fw_res = fw_model(*torch_inp_tensors)

        exec_net = core.compile_model(ir_test, device_name='GPU', config={"INFERENCE_PRECISION_HINT": "f16"})
        request = exec_net.create_infer_request()
        ov_res = request.infer(example_inputs)

        npt.assert_allclose(ov_res[0], fw_res.numpy(), atol=1e-3, rtol=1e-3)
