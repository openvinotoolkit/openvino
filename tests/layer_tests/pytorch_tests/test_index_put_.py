# Copyright (C) 2018-2025 Intel Corporation
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

        ref_net = None

        return aten_index_put_(indices, accumulate), ref_net, "aten::index_put_"

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
        self.input_tensor = np.random.randn(*input_data["input_shape"]).astype(np.float32)
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

        ref_net = None

        return aten_index_put_(indices, accumulate), ref_net, "aten::index_put_"

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
        self.input_tensor = np.random.randn(*input_data["input_shape"]).astype(np.float32)
        self.values = input_data["values"]
        self._test(*self.create_model(indices, accumulate), ie_device, precision, ir_version)


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

        ref_net = None

        return aten_index_put_(accumulate), ref_net, "aten::index_put_"

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
        self.input_tensor = np.random.randn(*input_data["input_shape"]).astype(np.float32)
        self.values = input_data["values"]
        for i in range(len(indices)):
            if type(indices[i]) is list:
                indices[i] = np.random.randint(0, 2, indices[i])
        self.indices_0 = indices[0]
        self.indices_1 = indices[1]
        self._test(*self.create_model(accumulate), ie_device, precision, ir_version, trace_model=True, use_convert_model=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_nonzero_index_put_different_ranks(self, ie_device, precision, ir_version):
        self.input_tensor = np.random.randn(1, 10, 2).astype(np.float32)
        self.values = np.zeros((10, 2), dtype=np.float32)
        self.indices_0 = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0]]).astype(np.float32)
        self.indices_1 = np.zeros((1, 10), dtype=np.float32)
        self._test(*self.create_model(False), ie_device, precision, ir_version, trace_model=True, use_convert_model=True)


class TestMask_IndexPut(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(100, 5).astype(np.float32), np.random.randn(100, 5).astype(np.float32))

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
        self._test(self.aten_index_put_mask(), None, "aten::index_put_", ie_device, precision,
                   ir_version, trace_model=True, use_convert_model=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_index_put_masked_fill(self, ie_device, precision, ir_version):
        self._test(self.aten_index_put_mask2(), None, "aten::index_put_", ie_device, precision,
                   ir_version, trace_model=True, use_convert_model=True)


class TestMaskKosmos_IndexPut(PytorchLayerTest):
    def _prepare_input(self):
        mask = np.random.randint(0, 2, [1, 30]).astype(np.bool_)
        num = mask.sum()
        return (np.random.randn(1, 30, 50).astype(np.float32), mask.astype(np.int32), np.random.randn(num, 50).astype(np.float32))

    def create_model(self):
        class aten_index_put_mask(torch.nn.Module):
            def forward(self, x, y, z):
                x[y.to(dtype=torch.bool)] = z
                return x

        ref_net = None

        return aten_index_put_mask(), ref_net, "aten::index_put_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_nonzero_kosmos_index_put_(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, trace_model=True, use_convert_model=True)
