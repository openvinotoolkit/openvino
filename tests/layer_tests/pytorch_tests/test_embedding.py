# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestEmbedding(PytorchLayerTest):
    def _prepare_input(self, indicies_size, indicies_dtype):
        return (self.random.randint(0, 9, size=indicies_size, dtype=indicies_dtype), self.random.randn(10, 10))

    def create_model(self):
        import torch
        import torch.nn.functional as F

        class aten_embedding(torch.nn.Module):

            def forward(self, indicies, weight):
                return F.embedding(indicies, weight)


        return aten_embedding(), "aten::embedding"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("indicies_size", [1, 2, 3, 4])
    @pytest.mark.parametrize("indicies_dtype", ["int", "int32"])
    def test_embedding(self, ie_device, precision, ir_version, indicies_size, indicies_dtype):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"indicies_size": indicies_size, "indicies_dtype": indicies_dtype})
