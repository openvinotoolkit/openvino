# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestSplitAmax(PytorchLayerTest):
    """Test case for split_with_sizes followed by amax operation.

    This test reproduces the issue where SequenceMark is not removed
    when a ReduceMax operation consumes the output of split_with_sizes.
    """

    def _prepare_input(self):
        return (self.random.randn(1, 3, 80),)

    def create_model(self):
        class SplitAmaxModel(torch.nn.Module):
            def forward(self, x):
                # Split the tensor along dimension 1
                splits = x.split([1, 1, 1], dim=1)
                # Apply amax to the first split
                result = splits[0].amax(dim=1)
                return result

        return SplitAmaxModel(), ["aten::split_with_sizes", "aten::amax"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_split_amax(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, trace_model=True)


class TestSplitAmaxWithoutGetitem(PytorchLayerTest):
    """Test case for split_with_sizes followed by amax without explicit getitem.

    This test case directly uses the split result with amax,
    simulating the YOLOv10 pattern mentioned in the issue.
    """

    def _prepare_input(self):
        return (self.random.randn(2, 5, 80),)

    def create_model(self):
        class SplitAmaxDirectModel(torch.nn.Module):
            def forward(self, x):
                # Split along the last dimension
                chunk1, chunk2 = x.split([2, 3], dim=1)
                # Apply amax to get maximum values
                max_vals = chunk1.amax(dim=-1)
                return max_vals

        return SplitAmaxDirectModel(), ["aten::split_with_sizes", "aten::amax"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_split_amax_direct(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, trace_model=True)


class TestSplitAmaxMultiDim(PytorchLayerTest):
    """Test case for split_with_sizes followed by amax with multiple dimensions.

    Tests that SequenceMark is correctly handled when amax operates on multiple dimensions.
    """

    def _prepare_input(self):
        return (self.random.randn(2, 4, 5, 6),)

    def create_model(self):
        class SplitAmaxMultiDimModel(torch.nn.Module):
            def forward(self, x):
                # Split along dimension 1
                chunk1, chunk2 = x.split([2, 2], dim=1)
                # Apply amax on multiple dimensions
                result = chunk1.amax(dim=[1, 2])
                return result

        return SplitAmaxMultiDimModel(), ["aten::split_with_sizes", "aten::amax"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_split_amax_multidim(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, trace_model=True)


class TestSplitAmin(PytorchLayerTest):
    """Test case for split_with_sizes followed by amin operation.

    Verifies that SequenceMark handling also works for amin (minimum reduction).
    """

    def _prepare_input(self):
        return (self.random.randn(2, 6, 40),)

    def create_model(self):
        class SplitAminModel(torch.nn.Module):
            def forward(self, x):
                # Split along dimension 1
                splits = x.split([2, 2, 2], dim=1)
                # Apply amin to the second split
                result = splits[1].amin(dim=1)
                return result

        return SplitAminModel(), ["aten::split_with_sizes", "aten::amin"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_split_amin(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, trace_model=True)


class TestSplitSum(PytorchLayerTest):
    """Test case for split_with_sizes followed by sum operation.

    Tests that SequenceMark handling works with sum reduction.
    """

    def _prepare_input(self):
        return (self.random.randn(3, 9, 20),)

    def create_model(self):
        class SplitSumModel(torch.nn.Module):
            def forward(self, x):
                # Split along dimension 1
                chunk1, chunk2, chunk3 = x.split([3, 3, 3], dim=1)
                # Apply sum with dimension parameter
                result = chunk2.sum(dim=[1, 2])
                return result

        return SplitSumModel(), ["aten::split_with_sizes", "aten::sum"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_split_sum(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, trace_model=True)


class TestSplitLogsumexp(PytorchLayerTest):
    """Test case for split_with_sizes followed by logsumexp operation.

    Verifies SequenceMark handling with logsumexp which internally uses ReduceMax.
    """

    def _prepare_input(self):
        return (self.random.randn(2, 4, 30),)

    def create_model(self):
        class SplitLogsumexpModel(torch.nn.Module):
            def forward(self, x):
                # Split along dimension 1
                chunk1, chunk2 = x.split([2, 2], dim=1)
                # Apply logsumexp with dimension parameter
                result = torch.logsumexp(chunk1, dim=1)
                return result

        return SplitLogsumexpModel(), ["aten::split_with_sizes", "aten::logsumexp"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_split_logsumexp(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, trace_model=True)
