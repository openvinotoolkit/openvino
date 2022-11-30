# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
import numpy as np
import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestFull(PytorchLayerTest):
    def _prepare_input(self, value):
        return (np.array(value, dtype=np.float32), )

    def create_model(self, shape):
        import torch

        class aten_full(torch.nn.Module):
            def __init__(self, shape):
                super(aten_full, self).__init__()
                self.shape = shape

            def forward(self, x: float):
                return torch.full(self.shape, x)

        ref_net = None

        return aten_full(shape), ref_net, "aten::full"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value", [0, 1, -1, 0.5])
    @pytest.mark.nightly
    def test_full(self, shape, value, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={'value': value})


class TestFullLike(PytorchLayerTest):
    def _prepare_input(self, value, shape):
        return (np.random.randn(*shape).astype(np.float32), np.array(value, dtype=np.float32), )

    def create_model(self):
        import torch

        class aten_full_like(torch.nn.Module):

            def forward(self, input_t: torch.Tensor, x: float):
                return torch.full_like(input_t, x)

        ref_net = None

        return aten_full_like(), ref_net, "aten::full_like"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value", [0, 1, -1, 0.5])
    @pytest.mark.nightly
    def test_full_like(self, shape, value, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'value': value, 'shape': shape})


class TestNewFull(PytorchLayerTest):
    def _prepare_input(self, value, input_dtype=np.float32):
        return (np.random.randn(1, 3, 10, 10).astype(input_dtype), np.array(value, dtype=np.float32))

    def create_model(self, shape):
        import torch

        class aten_full(torch.nn.Module):
            def __init__(self, shape):
                super(aten_full, self).__init__()
                self.shape = shape

            def forward(self, input_tensor: torch.Tensor, x: float):
                return input_tensor.new_full(self.shape, x)

        ref_net = None

        return aten_full(shape), ref_net, "aten::new_full"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value,input_dtype", [(0, np.uint8), (1, np.int32), (-1, np.float32), (0.5, np.float64)])
    @pytest.mark.nightly
    def test_new_full(self, shape, value, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'value': value, 'input_dtype': input_dtype})


class TestZerosAndOnes(PytorchLayerTest):
    def _prepare_input(self, shape):
        return (np.random.randn(*shape).astype(np.float32),)

    def create_model(self, op_type):
        import torch
        ops = {
            "aten::zeros": torch.zeros,
            "aten::ones": torch.ones,
            "aten::zeros_like": torch.zeros_like,
            "aten::ones_like": torch.ones_like
        }

        class aten_op(torch.nn.Module):
            def __init__(self, op):
                super(aten_op, self).__init__()
                self.op = op

            def forward(self, x):
                shape = x.shape
                return self.op(shape)

        class aten_op_like(torch.nn.Module):
            def __init__(self, op):
                super(aten_op_like, self).__init__()
                self.op = op

            def forward(self, x):
                return self.op(x)

        model_cls = aten_op_like if op_type.endswith('_like') else aten_op
        op = ops[op_type]

        ref_net = None

        return model_cls(op), ref_net, op_type

    @pytest.mark.parametrize("shape", [(1, 1), (1, 2), (1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5, 6)])
    @pytest.mark.parametrize("op_type", ["aten::zeros", "aten::ones", "aten::zeros_like", "aten::ones_like"])
    @pytest.mark.nightly
    def test_fill(self, op_type, shape, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={'shape': shape})


class TestNewZeros(PytorchLayerTest):
    def _prepare_input(self, input_dtype=np.float32):
        return (np.random.randn(1, 3, 10, 10).astype(input_dtype), )

    def create_model(self, shape):
        import torch

        class aten_full(torch.nn.Module):
            def __init__(self, shape):
                super(aten_full, self).__init__()
                self.shape = shape

            def forward(self, input_tensor: torch.Tensor):
                return input_tensor.new_zeros(self.shape)

        ref_net = None

        return aten_full(shape), ref_net, "aten::new_zeros"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.nightly
    def test_new_zeros(self, shape, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_dtype': input_dtype})


class TestNewOnes(PytorchLayerTest):
    def _prepare_input(self, input_dtype=np.float32):
        return (np.random.randn(1, 3, 10, 10).astype(input_dtype), )

    def create_model(self, shape):
        import torch

        class aten_full(torch.nn.Module):
            def __init__(self, shape):
                super(aten_full, self).__init__()
                self.shape = shape

            def forward(self, input_tensor: torch.Tensor):
                return input_tensor.new_ones(self.shape)

        ref_net = None

        return aten_full(shape), ref_net, "aten::new_ones"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.nightly
    def test_new_ones(self, shape, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_dtype': input_dtype})
