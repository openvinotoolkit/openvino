# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestFlatten(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 3, 4, 5),)

    def create_model(self, dim0, dim1):
        import torch

        class aten_flatten(torch.nn.Module):
            def __init__(self, dim0, dim1):
                super().__init__()
                self.dim0 = dim0
                self.dim1 = dim1

            def forward(self, x):
                return torch.flatten(x, self.dim0, self.dim1)


        return aten_flatten(dim0, dim1), "aten::flatten"

    @pytest.mark.parametrize("dim0,dim1", [[0, -1],
                                           [-2, -1],
                                           [0, 1],
                                           [0, 2],
                                           [0, 3],
                                           [1, 2],
                                           [1, 3],
                                           [2, 3]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_flatten(self, dim0, dim1, ie_device, precision, ir_version):
        self._test(*self.create_model(dim0, dim1),
                   ie_device, precision, ir_version)


class TestFlattenRankSensitiveConsumer(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 3, 4, 5),)

    def create_model(self):
        import torch

        class aten_flatten_permute(torch.nn.Module):
            def forward(self, x):
                x = torch.flatten(x, 1, 2)
                return torch.permute(x, (0, 2, 1))

        return aten_flatten_permute(), ["aten::flatten", "aten::permute"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_flatten_static_rank_dynamic_dims_before_permute(self, ie_device, precision, ir_version):
        self._test(*self.create_model(),
                   ie_device, precision, ir_version,
                   trace_model=True)
