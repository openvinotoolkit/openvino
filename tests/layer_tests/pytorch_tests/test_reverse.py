# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class aten_flip(torch.nn.Module):
    """Model that flips tensor along specified dimensions."""

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.flip(x, dims=self.dims)


class aten_reverse_list(torch.nn.Module):
    """Model that performs list reversal on a statically known list."""

    def forward(self, x):
        l = [x[0], x[1], x[2], x[3], x[4]]
        l.reverse()
        # Use stack for 0-dimensional tensors (indexing a 1D tensor gives 0-d tensors)
        return torch.stack(l, dim=0)


class TestReverse(PytorchLayerTest):
    """Test suite for flip and reverse operations."""

    def _prepare_input(self, input_shape=None, dtype=np.float32):
        if input_shape is None:
            input_shape = self.input_shape
        return (np.arange(np.prod(input_shape), dtype=dtype).reshape(input_shape),)

    @pytest.mark.parametrize(
        "input_shape",
        [[5], [3, 4], [2, 3, 4]],
    )
    @pytest.mark.parametrize("dim", [0, -1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_flip(self, ie_device, precision, ir_version, input_shape, dim):
        self.input_shape = input_shape
        self._test(
            aten_flip((dim,)),
            None,
            "aten::flip",
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape},
        )

    @pytest.mark.parametrize(
        "input_shape",
        [[5]],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_reverse_list(self, ie_device, precision, ir_version, input_shape):
        if ie_device == "GPU":
            pytest.skip(
                "List reverse with 0-d tensors causes OpenCL buffer issues on GPU"
            )
        self.input_shape = input_shape
        self._test(
            aten_reverse_list(),
            None,
            "aten::reverse",
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape},
        )
