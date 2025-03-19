# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestAsStrided(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(8, 8).astype(np.float32),)

    def create_model(self, size, stride, offset):
        class aten_as_strided(torch.nn.Module):
            def __init__(self, size, stride, offset):
                super().__init__()
                self.size = size
                self.stride = stride
                self.offset = offset

            def forward(self, x):
                return torch.as_strided(x, self.size, self.stride, self.offset)

        ref_net = None

        return aten_as_strided(size, stride, offset), ref_net, "aten::as_strided"

    @pytest.mark.parametrize(
        "size,stride",
        [
            ([1], [1]),
            ([2, 2], [1, 1]),
            ([5, 4, 3], [1, 3, 7]),
            ([5, 5, 5], [5, 0, 5]),
            ([1, 2, 3, 4], [4, 3, 2, 1]),
        ],
    )
    @pytest.mark.parametrize("offset", [None, 1, 3, 7])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_as_strided(self, size, stride, offset, ie_device, precision, ir_version):
        self._test(*self.create_model(size, stride, offset), ie_device, precision, ir_version, trace_model=True)

class TestAsStridedCopy(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(8, 8).astype(np.float32),)

    def create_model(self, size, stride, offset):
        class aten_as_strided_copy(torch.nn.Module):
            def __init__(self, size, stride, offset):
                super().__init__()
                self.size = size
                self.stride = stride
                self.offset = offset

            def forward(self, x):
                return torch.as_strided_copy(x, self.size, self.stride, self.offset)

        ref_net = None

        return aten_as_strided_copy(size, stride, offset), ref_net, "aten::as_strided_copy"

    @pytest.mark.parametrize(
        "size,stride",
        [
            ([1], [1]),
            ([2, 2], [1, 1]),
            ([5, 4, 3], [1, 3, 7]),
            ([5, 5, 5], [5, 0, 5]),
            ([1, 2, 3, 4], [4, 3, 2, 1]),
        ],
    )
    @pytest.mark.parametrize("offset", [None, 1, 3, 7])
    @pytest.mark.precommit_fx_backend
    def test_as_strided_copy(self, size, stride, offset, ie_device, precision, ir_version):
        self._test(*self.create_model(size, stride, offset), ie_device, precision, ir_version, trace_model=True)


class TestAsStridedListConstruct(PytorchLayerTest):
    def _prepare_input(self, size_shape_tensor=[1], stride_shape_tensor=[1]):
        return (
            np.random.randn(8, 8).astype(np.float32),
            np.ones(size_shape_tensor),
            np.ones(stride_shape_tensor),
        )

    def create_model(self, size, stride, offset, mode):
        class aten_as_strided(torch.nn.Module):
            def __init__(self, size, stride, offset, mode):
                super().__init__()
                self.size = size
                self.stride = stride
                self.size_shape_tensor = torch.empty(size)
                self.stride_shape_tensor = torch.empty(stride)
                self.offset = offset
                modes = {
                    "no_const": self.forward_no_const,
                    "stride_const": self.forward_stride_const,
                    "size_const": self.forward_size_const,
                }
                self.forward = modes.get(mode)

            def forward_no_const(self, x, size_shape_tensor, stride_shape_tensor):
                sz1, sz2, sz3 = size_shape_tensor.shape
                st1, st2, st3 = stride_shape_tensor.shape
                return torch.as_strided(x, [sz1, sz2, sz3], [st1, st2, st3], self.offset)

            def forward_stride_const(self, x, size_shape_tensor, stride_shape_tensor):
                sz1, sz2, sz3 = size_shape_tensor.shape
                return torch.as_strided(x, [sz1, sz2, sz3], self.stride, self.offset)

            def forward_size_const(self, x, size_shape_tensor, stride_shape_tensor):
                st1, st2, st3 = stride_shape_tensor.shape
                return torch.as_strided(x, self.size, [st1, st2, st3], self.offset)

        ref_net = None

        return aten_as_strided(size, stride, offset, mode), ref_net, ["aten::as_strided", "prim::ListConstruct"]

    @pytest.mark.parametrize("size,stride", [([5, 4, 3], [1, 3, 7]), ([5, 5, 5], [5, 0, 5])])
    @pytest.mark.parametrize("offset", [None, 7])
    @pytest.mark.parametrize("mode", ["no_const", "stride_const", "size_const"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_as_strided_list_construct(self, size, stride, offset, mode, ie_device, precision, ir_version):
        inp_kwargs = {"size_shape_tensor": size, "stride_shape_tensor": stride}
        self._test(
            *self.create_model(size, stride, offset, mode),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input=inp_kwargs,
            trace_model=True
        )


class TestAsStridedLongformer(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 10, 20, 40).astype(np.float32).transpose([0, 2, 3, 1]),)

    def create_model(self):
        class aten_as_strided_lf(torch.nn.Module):
            def forward(self, x):
                chunk_size = list(x.size())
                chunk_size[1] = chunk_size[1] * 2 - 1
                chunk_stride = list(x.stride())
                chunk_stride[1] = chunk_stride[1] // 2
                return x.as_strided(size=chunk_size, stride=chunk_stride)

        ref_net = None

        return aten_as_strided_lf(), ref_net, "aten::as_strided"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_as_strided_lf(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, trace_model=True, freeze_model=False)
