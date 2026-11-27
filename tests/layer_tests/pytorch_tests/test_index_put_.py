# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestIndexPut_SingleIndices(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.values)

    def create_model(self, indices, accumulate):
        class aten_index_put_(torch.nn.Module):
            def __init__(self, indices, accumulate):
                super().__init__()
                self.indices = indices
                self.accumulate = accumulate

            def forward(self, input_tensor, values):
                input_tensor.index_put_((self.indices,), values, self.accumulate)
                return input_tensor


        return aten_index_put_(indices, accumulate), "aten::index_put_"

    @pytest.mark.parametrize(
        "input_data",
        (
            {
                "input_shape": [5],
                "values": np.array(11).astype(np.float32)},
            {
                "input_shape": [3, 3],
                "values": np.array([10, 11, 12]).astype(np.float32),
            },
        ),
    )
    @pytest.mark.parametrize(
        "indices",
        (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([-1, -2], dtype=torch.long),
            torch.tensor([0, -1, -2], dtype=torch.long),
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([0, 1, 2], dtype=torch.long),
        ),
    )
    @pytest.mark.parametrize("accumulate", (True, False))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_index_put_single_indices(self, ie_device, precision, ir_version, input_data, indices, accumulate):
        self.input_tensor = self.random.randn(*input_data["input_shape"])
        self.values = input_data["values"]
        self._test(*self.create_model(indices, accumulate), ie_device, precision, ir_version)


class TestIndexPut_ManyIndices(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.values)

    def create_model(self, indices, accumulate):
        class aten_index_put_(torch.nn.Module):
            def __init__(self, indices, accumulate):
                super().__init__()
                self.indices_first = indices[0]
                self.indices_second = indices[1]
                self.accumulate = accumulate

            def forward(self, input_tensor, values):
                input_tensor.index_put_((self.indices_first, self.indices_second), values, self.accumulate)
                return input_tensor


        return aten_index_put_(indices, accumulate), "aten::index_put_"

    @pytest.mark.parametrize(
        "input_data",
        (
            {
                "input_shape": [3, 3],
                "values": np.array(12).astype(np.float32)
            },
            {
                "input_shape": [3, 3, 3],
                "values": np.array([10, 11, 12]).astype(np.float32),
            },
        ),
    )
    @pytest.mark.parametrize(
        "indices",
        (
            (torch.tensor([0], dtype=torch.long), torch.tensor([2], dtype=torch.long)),
            (torch.tensor([1, 2], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)),
            (torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)),
            (torch.tensor([0], dtype=torch.long), torch.tensor([-2], dtype=torch.long)),
            (torch.tensor([-1, -2], dtype=torch.long), torch.tensor([0, 1], dtype=torch.long)),
            (torch.tensor([0, -1], dtype=torch.long), torch.tensor([0, -1], dtype=torch.long)),
        ),
    )
    @pytest.mark.parametrize("accumulate", (True, False))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_index_put_many_indices(self, ie_device, precision, ir_version, input_data, indices, accumulate):
        self.input_tensor = self.random.randn(*input_data["input_shape"])
        self.values = input_data["values"]
        self._test(*self.create_model(indices, accumulate), ie_device, precision, ir_version)


class TestIndexPut_ManyIndicesWithNone(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.values)

    def create_model(self, indices, accumulate):
        class aten_index_put_(torch.nn.Module):
            def __init__(self, indices, accumulate):
                super().__init__()
                self.indices_first = indices[0]
                self.indices_second = indices[1]
                self.indices_third = indices[2]
                self.accumulate = accumulate

            def forward(self, input_tensor, values):
                return torch.ops.aten.index_put_.default(input_tensor, (self.indices_first, self.indices_second, self.indices_third), values, self.accumulate)


        return aten_index_put_(indices, accumulate), "aten::index_put_"

    @pytest.mark.parametrize(
        "input_data",
        (
            {
                "input_shape": [1, 8, 8, 8],
                "values": np.ones((8,1,8)).astype(np.float32)
            },
        ),
    )
    @pytest.mark.parametrize(
        "indices",
        (
            (None, None, torch.tensor([0], dtype=torch.long)),
        ),
    )
    @pytest.mark.parametrize("accumulate", (True, False))
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_index_put_many_indices_with_none(self, ie_device, precision, ir_version, input_data, indices, accumulate):
        self.input_tensor = np.zeros(input_data["input_shape"]).astype(np.float32)
        self.values = input_data["values"]
        self._test(*self.create_model(indices, accumulate), ie_device, precision, ir_version, aot_autograd=True)


class TestNonZero_IndexPut(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.values, self.indices_0, self.indices_1)

    def create_model(self, accumulate):
        class aten_index_put_(torch.nn.Module):
            def __init__(self, accumulate):
                super().__init__()
                self.accumulate = accumulate

            def forward(self, input_tensor, values, indices_0, indices_1):
                nonzero = (indices_0 == indices_1).nonzero(as_tuple=True)[0]
                input_tensor.index_put_((nonzero,), values, self.accumulate)
                return input_tensor


        return aten_index_put_(accumulate), "aten::index_put_"

    @pytest.mark.parametrize(
        "input_data",
        (
            {
                "input_shape": [3],
                "values": np.array(11).astype(np.float32),
            },
            {
                "input_shape": [3, 3],
                "values": np.array([10, 11, 12]).astype(np.float32),
            },
            {
                "input_shape": [3, 3, 3],
                "values": np.array([[10, 11, 12]]).astype(np.float32),
            },
            {
                "input_shape": [3, 3, 3],
                "values": np.array(10).astype(np.float32),
            },
            {
                "input_shape": [3, 3, 3],
                "values": np.zeros((1, 1, 3)).astype(np.float32),
            },
        ),
    )
    @pytest.mark.parametrize(
        "indices",
        (
            [[1, ], [1, ]],
            [[2, ], [2, ]],
            [np.array([0, 1, 0]), np.array([1, 1, 0])],
            [np.ones(shape=(3,)), np.ones(shape=(3,))],
            [np.ones(shape=(3,)), np.zeros(shape=(3,))],
        ),
    )
    @pytest.mark.parametrize("accumulate", (False, True))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_nonzero_index_put_(self, ie_device, precision, ir_version, input_data, indices, accumulate):
        self.input_tensor = self.random.randn(*input_data["input_shape"])
        self.values = input_data["values"]
        for i in range(len(indices)):
            if type(indices[i]) is list:
                indices[i] = self.random.randint(0, 2, indices[i])
        self.indices_0 = indices[0]
        self.indices_1 = indices[1]
        self._test(*self.create_model(accumulate), ie_device, precision, ir_version, trace_model=True, use_convert_model=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_nonzero_index_put_different_ranks(self, ie_device, precision, ir_version):
        self.input_tensor = self.random.randn(1, 10, 2)
        self.values = np.zeros((10, 2), dtype=np.float32)
        self.indices_0 = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0]]).astype(np.float32)
        self.indices_1 = np.zeros((1, 10), dtype=np.float32)
        self._test(*self.create_model(False), ie_device, precision, ir_version, trace_model=True, use_convert_model=True)


class TestMask_IndexPut(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(100, 5), self.random.randn(100, 5))

    class aten_index_put_mask(torch.nn.Module):
        def forward(self, x, y):
            x[x < 0] = y[x < 0]
            return x

    class aten_index_put_mask2(torch.nn.Module):
        def forward(self, x, y):
            x[x < 0] = 0
            return x

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_nonzero_index_put_(self, ie_device, precision, ir_version):
        self._test(self.aten_index_put_mask(), "aten::index_put_", ie_device, precision,
                   ir_version, trace_model=True, use_convert_model=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_index_put_masked_fill(self, ie_device, precision, ir_version):
        self._test(self.aten_index_put_mask2(), "aten::index_put_", ie_device, precision,
                   ir_version, trace_model=True, use_convert_model=True)


class TestIndexPut_LeadingNoneScalarValue(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(*self.shape),)

    def create_model(self, idx, scalar_value):
        class aten_index_put_flip(torch.nn.Module):
            def forward(self, x):
                x = x.clone()
                if scalar_value:
                    # Keep index_put in the graph: Python scalar assignment exports
                    # as index_fill or masked_fill in recent PyTorch versions.
                    x = torch.ops.aten.index_put_.default(
                        x, [None, torch.as_tensor(idx[1])], torch.tensor(2.0))
                else:
                    x[idx] = 1.0 - x[idx]
                return x

        return aten_index_put_flip(), "aten::index_put_"

    @pytest.mark.parametrize("idx", [
        (Ellipsis, [7]),
        (slice(None), [7]),
        (slice(None), [2, 7]),
        (Ellipsis, torch.arange(8) == 7),
    ])
    @pytest.mark.parametrize("shape", [[1, 8], [2, 8]])
    @pytest.mark.parametrize("scalar_value", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_index_put_leading_none_scalar_value(self, ie_device, precision, ir_version, idx, shape, scalar_value):
        self.shape = shape
        self._test(*self.create_model(idx, scalar_value), ie_device, precision, ir_version, trace_model=True)


class TestIndexPut_IndexDtypes(PytorchLayerTest):
    def _prepare_input(self):
        return (
            self.random.randn(4, 8).astype(np.float32),
            np.array([[0], [-2]], dtype=self.row_dtype),
            np.array([[1, -1, 3]], dtype=self.col_dtype),
            np.array(2.0, dtype=np.float32),
        )

    def create_model(self, accumulate):
        class aten_index_put(torch.nn.Module):
            def forward(self, x, rows, cols, values):
                x = x.clone()
                return x.index_put_((rows, cols), values, accumulate=accumulate)

        return aten_index_put(), "aten::index_put_"

    @pytest.mark.parametrize("row_dtype", [np.int32, np.int64])
    @pytest.mark.parametrize("col_dtype", [np.int32, np.int64])
    @pytest.mark.parametrize("accumulate", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_index_put_index_dtypes(self, ie_device, precision, ir_version, row_dtype, col_dtype, accumulate):
        if ie_device == "GPU" and accumulate:
            pytest.skip("GPU does not support ScatterNDUpdate with SUM reduction")
        self.row_dtype = row_dtype
        self.col_dtype = col_dtype
        self._test(*self.create_model(accumulate), ie_device, precision, ir_version, trace_model=True)


MASK_1D = [False, True, False]
# 5 selected elements
MASK_2D = [[True, False, True, False], [False, False, True, True], [True, False, False, False]]


class TestIndexPut_MaskAccumulate(PytorchLayerTest):
    def _prepare_input(self):
        x = self.random.randn(*self.input_shape)
        values = np.array(self.random.randn(*self.value_shape))
        if self.dtype == np.bool_:
            # For booleans, accumulate means logical or.
            return (x > 0, np.asarray(values > 0))
        return (x.astype(self.dtype), values.astype(self.dtype))

    def create_model(self, mask, leading_none, accumulate):
        class aten_index_put_mask(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mask = torch.tensor(mask)

            def forward(self, x, values):
                x = x.clone()
                indices = [None, self.mask] if leading_none else [self.mask]
                return torch.ops.aten.index_put_.default(x, indices, values, accumulate)

        return aten_index_put_mask(), "aten::index_put_"

    @pytest.mark.parametrize("input_shape,mask,leading_none,value_shape", [
        ([2, 3], MASK_1D, True, []),
        ([2, 3], MASK_1D, True, [2, 1]),
        ([2, 3, 4], MASK_2D, True, []),
        ([2, 3, 4], MASK_2D, True, [2, 5]),
        ([2, 3, 4], MASK_2D, True, [5]),
        ([2, 3, 4], MASK_2D, True, [2, 1]),
        ([2, 3, 4], MASK_2D, True, [1]),
        ([3, 4, 2], MASK_2D, False, []),
        ([3, 4, 2], MASK_2D, False, [5, 2]),
        ([3, 4, 2], MASK_2D, False, [2]),
    ])
    @pytest.mark.parametrize("accumulate", [False, True])
    @pytest.mark.parametrize("dtype", [np.float32, np.bool_])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_index_put_mask_accumulate(self, ie_device, precision, ir_version,
                                       input_shape, mask, leading_none, value_shape, accumulate, dtype):
        if ie_device == "GPU" and accumulate and value_shape:
            pytest.skip("GPU does not support ScatterNDUpdate with SUM reduction")
        self.input_shape = input_shape
        self.value_shape = value_shape
        self.dtype = dtype
        self._test(*self.create_model(mask, leading_none, accumulate),
                   ie_device, precision, ir_version, trace_model=True)


class TestMaskKosmos_IndexPut(PytorchLayerTest):
    def _prepare_input(self):
        mask = self.random.randint(0, 2, [1, 30], dtype=np.bool_)
        num = mask.sum()
        return (self.random.randn(1, 30, 50), mask.astype(np.int32), self.random.randn(num, 50))

    def create_model(self):
        class aten_index_put_mask(torch.nn.Module):
            def forward(self, x, y, z):
                x[y.to(dtype=torch.bool)] = z
                return x


        return aten_index_put_mask(), "aten::index_put_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_nonzero_kosmos_index_put_(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, trace_model=True, use_convert_model=True)
