# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize(
    "start, stop, step",
    [
        [1, 32, 1],
        [1, 32, 2],
        [1, 32, 10],
        [1, 32, -1],
        [1, 32, -2],
        [1, 32, -10],
        [32, 1, -1],
        [32, 1, -2],
        [32, 1, -10],
        [32, -31, -1],
        [32, -31, -2],
        [32, -31, -10],
    ],
)
class TestDeriveIndexRangeLength(PytorchLayerTest):
    def _prepare_input(self):
        input_data = np.array([self.start, self.stop, self.step])
        return (input_data,)

    def create_model(self):
        class prim_derive_index_range_length(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                start = int(x[0])
                stop = int(x[1])
                step = int(x[2])
                accumulator = 0
                for idx in range(start, stop, step):
                    accumulator += idx
                return accumulator

        ref_net = None

        return prim_derive_index_range_length(), ref_net, ["aten::__range_length", "aten::__derive_index"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_derive_index_range_length(self, ie_device, precision, ir_version, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step
        if ((stop - start) / step) < 0:
            pytest.xfail("Failed due to prim::Loop translation not supporting 0 iterations. Ticket: 110808")
        self._test(*self.create_model(), ie_device, precision, ir_version, freeze_model=False, trace_model=False)
