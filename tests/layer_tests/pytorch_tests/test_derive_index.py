# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestDeriveIndex(PytorchLayerTest):
    def _prepare_input(self):
        input_data = np.random.randint(0, 2, (64)) > 0
        return (input_data,)

    def create_model(self, start, stop, step):
        class prim_derive_index(torch.nn.Module):
            def __init__(self, start, stop, step) -> None:
                super().__init__()
                self.start = start
                self.stop = stop
                self.step = step
                self.a = torch.tensor([1])
                self.b = torch.tensor([0])

            def forward(self, x):
                tensor_list = []
                for idx in range(self.start, self.stop, self.step):
                    tensor_list.append(torch.where(x[idx], self.a, self.b))
                return torch.cat(tensor_list)

        ref_net = None

        return prim_derive_index(start, stop, step), ref_net, "aten::__derive_index"

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
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_derive_index(self, ie_device, precision, ir_version, start, stop, step):
        if ((stop - start) / step) < 0:
            pytest.xfail("Failed due to prim::Loop translation not supporting 0 iterations. Ticket: 110808")
        self._test(*self.create_model(start, stop, step), ie_device, precision, ir_version, freeze_model=False)
