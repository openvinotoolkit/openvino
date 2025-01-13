# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestEmptyNumeric(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(10, 10, 10),)

    def create_model(self, dtype):

        class aten_empty(torch.nn.Module):

            def __init__(self, dtype) -> None:
                dtype_map = {
                    "float32": torch.float32,
                    "float64": torch.float64,
                    "int64": torch.int64,
                    "int32": torch.int32,
                    "uint8": torch.uint8,
                    "int8": torch.int8
                }
                super().__init__()
                self.dtype = dtype_map[dtype]

            def forward(self, input_tensor):
                size = input_tensor.shape
                empty = torch.empty(size, dtype=self.dtype)
                # We don't want to compare values, just shape and type,
                # so we call zeros_like on data. Multiplying by zero would
                # produce sporadic errors if nan would be in empty.
                return torch.zeros_like(empty)

        ref_net = None

        return aten_empty(dtype), ref_net, "aten::empty"

    @pytest.mark.parametrize('dtype', ("float32", "float64", "int64", "int32", "uint8", "int8"))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_empty(self, ie_device, precision, ir_version, dtype):
        self._test(*self.create_model(dtype), ie_device, precision, ir_version)


class TestEmptyLike(PytorchLayerTest):

    def _prepare_input(self, shape, dtype=np.float32, out=False):
        if not out:
            return (np.random.randn(*shape).astype(dtype if dtype is not None else np.float32),)
        return (np.random.randn(*shape), np.ones(shape, dtype=(dtype if dtype is not None else np.float32)))

    def create_model(self, dtype, out, no_expose_dtype=False):

        class aten_empty_like(torch.nn.Module):
                
            def __init__(self, dtype=None, out=False, no_expose_dtype=False):
                dtype_map = {
                    "float32": torch.float32,
                    "float64": torch.float64,
                    "int64": torch.int64,
                    "int32": torch.int32,
                    "uint8": torch.uint8,
                    "int8": torch.int8
                }
                super().__init__()
                self.dtype = dtype_map.get(dtype, None)
                if out:
                    self.forward = self.forward_out
                if no_expose_dtype:
                    self.forward = self.forward_input_dtype

            def forward(self, input_tensor):
                empty = torch.empty_like(input_tensor, dtype=self.dtype)
                # We don't want to compare values, just shape and type,
                # so we call zeros_like on data. Multiplying by zero would
                # produce sporadic errors if nan would be in empty.
                return torch.zeros_like(empty)

            def forward_input_dtype(self, input_tensor):
                # We don't want to compare values, just shape and type,
                # so we call zeros_like on data. Multiplying by zero would
                # produce sporadic errors if nan would be in empty.
                input_tensor.to(self.dtype)
                empty = torch.empty_like(input_tensor)
                return torch.zeros_like(empty)

            def forward_out(self, input_tensor, out_tensor):
                torch.empty_like(input_tensor, out=out_tensor)
                # We don't want to compare values, just shape and type,
                # so we call zeros_like on data. Multiplying by zero would
                # produce sporadic errors if nan would be in empty.
                return torch.zeros_like(out_tensor)

        ref_net = None

        return aten_empty_like(dtype, out, no_expose_dtype), ref_net, "aten::empty_like"

    @pytest.mark.parametrize('dtype', (None, "float32", "float64", "int64", "int32", "uint8", "int8"))
    @pytest.mark.parametrize("input_shape", [[2,], [1, 10], [10, 5, 2]])
    @pytest.mark.parametrize("out", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_empty_like(self, ie_device, precision, ir_version, dtype, input_shape, out):
        self._test(*self.create_model(dtype, out), ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"shape": input_shape, "out": out, "dtype": dtype})

    @pytest.mark.parametrize('dtype', (None, "float32", "float64", "int64", "int32", "uint8", "int8"))
    @pytest.mark.parametrize("input_shape", [[2,], [1, 10], [10, 5, 2]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_empty_like_no_dtype(self, ie_device, precision, ir_version, dtype, input_shape):
        self._test(*self.create_model(dtype, out=False, no_expose_dtype=True), ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"shape": input_shape, "out": False, "dtype": dtype})


class TestEmptyBoolean(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(10, 10, 10),)

    def create_model(self):

        class aten_empty(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.false = torch.tensor([False])

            def forward(self, input_tensor):
                size = input_tensor.shape
                empty = torch.empty(size, dtype=torch.bool)
                # We don't want to compare values, just shape and type,
                # so we do "and" operation with False.
                return empty & self.false

        ref_net = None

        return aten_empty(), ref_net, "aten::empty"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_empty_bool(self, ie_device, precision, ir_version, ):
        self._test(*self.create_model(), ie_device, precision, ir_version)

class TestNewEmpty(PytorchLayerTest):
    def _prepare_input(self, input_dtype=np.float32):
        return (np.random.randn(1, 3, 10, 10).astype(input_dtype),)

    def create_model(self, shape, dtype=None, used_dtype=False):
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "bool": torch.bool
        }

        class aten_empty(torch.nn.Module):
            def __init__(self, shape):
                super(aten_empty, self).__init__()
                self.shape = shape
                self.zero = torch.tensor([0])

            def forward(self, input_tensor: torch.Tensor):
                empty = input_tensor.new_empty(self.shape)
                # We don't want to compare values, just shape and type,
                # so we call zeros_like on data. Multiplying by zero would
                # produce sporadic errors if nan would be in empty.
                return torch.zeros_like(empty)

        class aten_empty_with_dtype(torch.nn.Module):
            def __init__(self, shape, dtype):
                super(aten_empty_with_dtype, self).__init__()
                self.shape = shape
                self.dtype = dtype
                self.zero = torch.tensor([0], dtype=self.dtype)

            def forward(self, input_tensor: torch.Tensor):
                empty = input_tensor.new_empty(self.shape, dtype=self.dtype)
                # We don't want to compare values, just shape and type,
                # so we call zeros_like on data. Multiplying by zero would
                # produce sporadic errors if nan would be in empty.
                return torch.zeros_like(empty)

        ref_net = None
        model = aten_empty(shape)

        if used_dtype:
            dtype = dtype_map[dtype]
            model = aten_empty_with_dtype(shape, dtype)

        return model, ref_net, "aten::new_empty"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_new_empty(self, shape, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_dtype': input_dtype}, use_convert_model=True)

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("input_dtype", [bool, np.uint8, np.int8, np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.parametrize("dtype", ["bool", "uint8", "int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.nightly
    def test_new_empty_with_dtype(self, shape, dtype, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape, dtype=dtype, used_dtype=True), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_dtype': input_dtype}, use_convert_model=True)
