# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestFull(PytorchLayerTest):
    def _prepare_input(self, value):
        return (np.array(value, dtype=np.float32),)

    def create_model(self, shape, dtype=None, use_dtype=False, use_out=False, with_names=False):
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

        class aten_full(torch.nn.Module):
            def __init__(self, shape):
                super(aten_full, self).__init__()
                self.shape = shape

            def forward(self, x: float):
                return torch.full(self.shape, x)

        class aten_full_dtype(torch.nn.Module):
            def __init__(self, shape, dtype):
                super(aten_full_dtype, self).__init__()
                self.shape = shape
                self.dtype = dtype

            def forward(self, x: float):
                return torch.full(self.shape, x, dtype=self.dtype)

        class aten_full_dtype_with_names(torch.nn.Module):
            def __init__(self, shape, dtype):
                super(aten_full_dtype_with_names, self).__init__()
                self.shape = shape
                self.dtype = dtype

            def forward(self, x: float):
                return torch.full(self.shape, x, dtype=self.dtype, names=None)

        class aten_full_out(torch.nn.Module):
            def __init__(self, shape, dtype):
                super(aten_full_out, self).__init__()
                self.shape = shape
                self.dtype = dtype

            def forward(self, x: float):
                return torch.full(self.shape, x, out=torch.tensor(1, dtype=self.dtype))

        class aten_full_out_with_names(torch.nn.Module):
            def __init__(self, shape, dtype):
                super(aten_full_out_with_names, self).__init__()
                self.shape = shape
                self.dtype = dtype

            def forward(self, x: float):
                return torch.full(self.shape, x, out=torch.tensor(1, dtype=self.dtype), names=None)

        ref_net = None
        model = aten_full(shape)
        if use_dtype or use_out:
            dtype = dtype_map.get(dtype, dtype)
            if not use_out:
                model = aten_full_dtype(shape, dtype) if not with_names else aten_full_dtype_with_names(shape, dtype)
            else:
                model = aten_full_out(shape, dtype) if not with_names else aten_full_out_with_names(shape, dtype)

        return model, ref_net, "aten::full"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value", [0, 1, -1, 0.5])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_full(self, shape, value, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={'value': value})

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value", [0, 1, -1, 0.5])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("with_names", [skip_if_export(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit_fx_backend
    def test_full_dtype(self, shape, value, dtype, with_names, ie_device, precision, ir_version):
        self._test(*self.create_model(shape, dtype=dtype, use_dtype=True, with_names=with_names), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={'value': value})

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value", [0, 1, -1, 0.5])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("with_names", [skip_if_export(True), False])
    @pytest.mark.nightly
    def test_full_out(self, shape, value, dtype, with_names, ie_device, precision, ir_version):
        self._test(*self.create_model(shape, dtype=dtype, use_out=True, with_names=with_names), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={'value': value})

class TestFill(PytorchLayerTest):
    def _prepare_input(self, value, shape, input_dtype, value_dtype, out=False):
        if not out:
            return (np.random.randn(*shape).astype(input_dtype), np.array(value, dtype=value_dtype),)
        return (np.random.randn(*shape).astype(input_dtype), np.array(value, dtype=value_dtype), np.zeros(shape, dtype=input_dtype))


    def create_model(self, mode):
        import torch

        class aten_fill(torch.nn.Module):
            def __init__(self, mode) -> None:
                super().__init__()
                if mode == "inplace":
                    self.forward = self.forward_inplace
                if mode == "out":
                    self.forward = self.forward_out


            def forward_inplace(self, input_t: torch.Tensor, x: float):
                return input_t.fill_(x)
        
            def forward_out(self, input_t: torch.Tensor, x: float, out: torch.Tensor):
                return input_t.fill(x, out=out), out
            
            def forward(self, input_t: torch.Tensor, x:float):
                return input_t.fill(x)
    
        ref_net = None

        model = aten_fill(mode)

        return model, ref_net, "aten::fill_" if mode == "inplace" else "aten::fill"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value", [0, 1, -1, 0.5])
    @pytest.mark.parametrize("input_dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("value_dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("mode", ["", "inplace", "out"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_fill(self, shape, value, input_dtype, value_dtype, mode, ie_device, precision, ir_version):
        self._test(*self.create_model(mode), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       'value': value, 
                       'shape': shape, 
                       "input_dtype": input_dtype, 
                       "value_dtype": value_dtype,
                       "out": mode == "out"
                       })

class TestFillDiagonal(PytorchLayerTest):
    def _prepare_input(self, shape, input_dtype, value, value_dtype):
        return np.zeros(shape).astype(input_dtype), np.array(value, dtype=value_dtype)

    def create_model(self, shape, wrap):
        import torch

        class aten_fill_diagonal(torch.nn.Module):
            def __init__(self, input_shape, wrap=False) -> None:
                super().__init__()
                self.wrap = wrap
                self.input_shape = input_shape

            def forward(self, x:torch.Tensor, y:float):
                x = x.reshape(self.input_shape)
                return x.fill_diagonal_(y, wrap=self.wrap), x
        
        ref_net = None

        model = aten_fill_diagonal(shape, wrap)
        return model, "aten::fill_diagonal_", ref_net

    @pytest.mark.parametrize("shape", ([4, 4], [5, 4], [8, 4], [4, 3],  [5, 5, 5], [3, 3, 3, 3], [4, 4, 4, 4, 4]))
    @pytest.mark.parametrize("value", [0, 1, -1, 2.5])
    @pytest.mark.parametrize("input_dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("value_dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("wrap", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_fill_diagonal(self, shape, value, input_dtype, value_dtype, wrap, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.xfail(reason="fill_diagonal is not supported on GPU")
        self._test(*self.create_model(shape, wrap), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       'value': value, 
                       'shape': shape, 
                       "input_dtype": input_dtype, 
                       "value_dtype": value_dtype
                       })


class TestZero(PytorchLayerTest):
    def _prepare_input(self, shape, input_dtype):
        return (np.random.randn(*shape).astype(input_dtype),)

    def create_model(self):
        import torch

        class aten_zero(torch.nn.Module):

            def forward(self, input_t: torch.Tensor):
                return input_t.zero_()
        ref_net = None

        model = aten_zero()

        return model, ref_net, "aten::zero_"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("input_dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_zero(self, shape, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'shape': shape, "input_dtype": input_dtype})

class TestFullLike(PytorchLayerTest):
    def _prepare_input(self, value, shape):
        return (np.random.randn(*shape).astype(np.float32), np.array(value, dtype=np.float32),)

    def create_model(self, dtype=None, use_dtype=False, use_out=False):
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

        class aten_full_like(torch.nn.Module):

            def forward(self, input_t: torch.Tensor, x: float):
                return torch.full_like(input_t, x)

        class aten_full_like_dtype(torch.nn.Module):
            def __init__(self, dtype):
                super(aten_full_like_dtype, self).__init__()
                self.dtype = dtype

            def forward(self, input_t: torch.Tensor, x: float):
                return torch.full_like(input_t, x, dtype=self.dtype)

        class aten_full_like_out(torch.nn.Module):
            def __init__(self, dtype):
                super(aten_full_like_out, self).__init__()
                self.dtype = dtype

            def forward(self, input_t: torch.Tensor, x: float):
                return torch.full_like(input_t, x, out=torch.tensor(1, dtype=self.dtype))

        ref_net = None

        model = aten_full_like()
        if use_dtype or use_out:
            dtype = dtype_map.get(dtype, dtype)
            if not use_out:
                model = aten_full_like_dtype(dtype)
            else:
                model = aten_full_like_out(dtype)

        return model, ref_net, "aten::full_like"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value", [0, 1, -1, 0.5])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_full_like(self, shape, value, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'value': value, 'shape': shape})

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value", [0, 1, -1, 0.5])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.nightly
    def test_full_like_dtype(self, shape, value, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, use_dtype=True), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'value': value, 'shape': shape})

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value", [0, 1, -1, 0.5])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.nightly
    def test_full_like_out(self, shape, value, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, use_out=True), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'value': value, 'shape': shape})


class TestNewFull(PytorchLayerTest):
    def _prepare_input(self, value, input_dtype=np.float32):
        return (np.random.randn(1, 3, 10, 10).astype(input_dtype), np.array(value, dtype=np.float32))

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

        class aten_full(torch.nn.Module):
            def __init__(self, shape):
                super(aten_full, self).__init__()
                self.shape = shape

            def forward(self, input_tensor: torch.Tensor, x: float):
                return input_tensor.new_full(self.shape, x), input_tensor

        class aten_full_with_dtype(torch.nn.Module):
            def __init__(self, shape, dtype):
                super(aten_full_with_dtype, self).__init__()
                self.shape = shape
                self.dtype = dtype

            def forward(self, input_tensor: torch.Tensor, x: float):
                return input_tensor.new_full(size=self.shape, fill_value=x, dtype=self.dtype), input_tensor

        ref_net = None
        model = aten_full(shape)

        if used_dtype:
            dtype = dtype_map[dtype]
            model = aten_full_with_dtype(shape, dtype)

        return model, ref_net, "aten::new_full"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value,input_dtype", [(0, np.uint8), (1, np.int32), (-1, np.float32), (0.5, np.float64)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_new_full(self, shape, value, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'value': value, 'input_dtype': input_dtype}, use_convert_model=True)

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("value,input_dtype", [(0, np.uint8), (1, np.int32), (-1, np.float32), (0.5, np.float64)])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.nightly
    def test_new_full_with_dtype(self, value, shape, dtype, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape, dtype=dtype, used_dtype=True), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'value': value, 'input_dtype': input_dtype}, use_convert_model=True)


class TestZerosAndOnes(PytorchLayerTest):
    def _prepare_input(self, shape):
        return (np.random.randn(*shape).astype(np.float32),)

    def create_model(self, op_type, dtype=None, with_dtype=False, with_out=False, with_names=False):
        import torch
        ops = {
            "aten::zeros": torch.zeros,
            "aten::ones": torch.ones,
            "aten::zeros_like": torch.zeros_like,
            "aten::ones_like": torch.ones_like
        }
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "bool": torch.bool
        }

        class aten_op(torch.nn.Module):
            def __init__(self, op):
                super(aten_op, self).__init__()
                self.op = op

            def forward(self, x):
                shape = x.shape
                return self.op(shape), x

        class aten_op_like(torch.nn.Module):
            def __init__(self, op):
                super(aten_op_like, self).__init__()
                self.op = op

            def forward(self, x):
                return self.op(x)

        class aten_op_dtype(torch.nn.Module):
            def __init__(self, op, dtype):
                super(aten_op_dtype, self).__init__()
                self.op = op
                self.dtype = dtype

            def forward(self, x):
                shape = x.shape
                return self.op(shape, dtype=self.dtype), x

        class aten_op_dtype_with_names(aten_op_dtype):
            def forward(self, x):
                shape = x.shape
                return self.op(shape, dtype=self.dtype, names=None), x

        class aten_op_like_dtype(torch.nn.Module):
            def __init__(self, op, dtype):
                super(aten_op_like_dtype, self).__init__()
                self.op = op
                self.dtype = dtype

            def forward(self, x):
                return self.op(x, dtype=self.dtype)

        class aten_op_out(torch.nn.Module):
            def __init__(self, op, dtype):
                super(aten_op_out, self).__init__()
                self.op = op
                self.dtype = dtype

            def forward(self, x):
                shape = x.shape
                return self.op(shape, out=torch.tensor(0, dtype=self.dtype))

        class aten_op_out_with_names(torch.nn.Module):
            def __init__(self, op, dtype):
                super(aten_op_out_with_names, self).__init__()
                self.op = op
                self.dtype = dtype

            def forward(self, x):
                shape = x.shape
                return self.op(shape, out=torch.tensor(0, dtype=self.dtype), names=None)

        class aten_op_like_out(torch.nn.Module):
            def __init__(self, op, dtype):
                super(aten_op_like_out, self).__init__()
                self.op = op
                self.dtype = dtype

            def forward(self, x):
                return self.op(x, out=torch.tensor(0, dtype=self.dtype))

        like = op_type.endswith('_like')
        op = ops[op_type]
        if not like:
            model_cls = aten_op(op)
            if with_dtype or with_out:
                dtype = dtype_map[dtype]
                if with_dtype:
                    model_cls = aten_op_dtype(op, dtype) if not with_names else aten_op_dtype_with_names(op, dtype)
                if with_out:
                    model_cls = aten_op_out(op, dtype) if not with_names else aten_op_out_with_names(op, dtype)
        else:
            model_cls = aten_op_like(op)
            if with_dtype or with_out:
                dtype = dtype_map[dtype]
                model_cls = aten_op_like_dtype(op, dtype) if not with_out else aten_op_like_out(op, dtype)

        ref_net = None

        return model_cls, ref_net, op_type

    @pytest.mark.parametrize("shape", [(1, 1), (1, 2), (1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5, 6)])
    @pytest.mark.parametrize("op_type", ["aten::zeros", "aten::ones", "aten::zeros_like", "aten::ones_like"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_zeros_ones(self, op_type, shape, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={'shape': shape})

    @pytest.mark.parametrize("shape", [(1, 1), (1, 2), (1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5, 6)])
    @pytest.mark.parametrize("op_type", ["aten::zeros", "aten::ones"])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("with_names", [skip_if_export(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    def test_zeros_ones_with_dtype(self, op_type, shape, dtype, with_names, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type, dtype=dtype, with_dtype=True, with_names=with_names), ie_device,
                   precision,
                   ir_version, kwargs_to_prepare_input={'shape': shape})

    @pytest.mark.parametrize("shape", [(1, 1), (1, 2), (1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5, 6)])
    @pytest.mark.parametrize("op_type", ["aten::zeros", "aten::ones"])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("with_names", [skip_if_export(True), False])
    @pytest.mark.nightly
    def test_zeros_ones_with_out(self, op_type, shape, dtype, with_names, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type, dtype=dtype, with_out=True, with_names=with_names), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={'shape': shape})

    @pytest.mark.parametrize("shape", [(1, 1), (1, 2), (1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5, 6)])
    @pytest.mark.parametrize("op_type", ["aten::zeros_like", "aten::ones_like"])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.nightly
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    def test_zeros_ones_like_with_dtype(self, op_type, shape, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type, dtype=dtype, with_dtype=True), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={'shape': shape})

    @pytest.mark.parametrize("shape", [(1, 1), (1, 2), (1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5, 6)])
    @pytest.mark.parametrize("op_type", ["aten::zeros_like", "aten::ones_like"])
    @pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.nightly
    def test_zeros_ones_like_with_out(self, op_type, shape, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type, dtype=dtype, with_out=True), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={'shape': shape})


class TestNewZeros(PytorchLayerTest):
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

        class aten_full(torch.nn.Module):
            def __init__(self, shape):
                super(aten_full, self).__init__()
                self.shape = shape

            def forward(self, input_tensor: torch.Tensor):
                return input_tensor.new_zeros(self.shape), input_tensor

        class aten_full_with_dtype(torch.nn.Module):
            def __init__(self, shape, dtype):
                super(aten_full_with_dtype, self).__init__()
                self.shape = shape
                self.dtype = dtype

            def forward(self, input_tensor: torch.Tensor):
                return input_tensor.new_zeros(self.shape, dtype=self.dtype), input_tensor

        ref_net = None
        model = aten_full(shape)

        if used_dtype:
            dtype = dtype_map[dtype]
            model = aten_full_with_dtype(shape, dtype)

        return model, ref_net, "aten::new_zeros"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_new_zeros(self, shape, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_dtype': input_dtype}, use_convert_model=True)

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("input_dtype", [bool, np.uint8, np.int8, np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.parametrize("dtype", ["bool", "uint8", "int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_new_zeros_with_dtype(self, shape, dtype, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape, dtype=dtype, used_dtype=True), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_dtype': input_dtype}, use_convert_model=True)


class TestNewOnes(PytorchLayerTest):
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

        class aten_full(torch.nn.Module):
            def __init__(self, shape):
                super(aten_full, self).__init__()
                self.shape = shape

            def forward(self, input_tensor: torch.Tensor):
                return input_tensor.new_ones(self.shape), input_tensor

        class aten_full_with_dtype(torch.nn.Module):
            def __init__(self, shape, dtype):
                super(aten_full_with_dtype, self).__init__()
                self.shape = shape
                self.dtype = dtype

            def forward(self, input_tensor: torch.Tensor):
                return input_tensor.new_ones(self.shape, dtype=self.dtype), input_tensor

        ref_net = None
        model = aten_full(shape)

        if used_dtype:
            dtype = dtype_map[dtype]
            model = aten_full_with_dtype(shape, dtype)

        return model, ref_net, "aten::new_ones"

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("input_dtype", [np.uint8, np.int8, np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_new_ones(self, shape, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_dtype': input_dtype}, use_convert_model=True)

    @pytest.mark.parametrize("shape", [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("input_dtype", [bool, np.uint8, np.int8, np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.parametrize("dtype", ["bool", "uint8", "int8", "int32", "int64", "float32", "float64"])
    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_new_ones_with_dtype(self, shape, dtype, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(shape, dtype=dtype, used_dtype=True), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_dtype': input_dtype}, use_convert_model=True)
