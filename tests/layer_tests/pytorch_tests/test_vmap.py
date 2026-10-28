# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestVmap(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(4, 6).astype("float32"), self.random.randint(0, 4, (3,)).astype("int64"))

    def create_model(self):
        class vmap_index(torch.nn.Module):
            # transformers' create_causal_mask builds the mask this way; torch.export keeps
            # vmap and indexing inside it as predispatch ops and flat_apply
            def forward(self, x, idx):
                return torch.vmap(lambda i: x[i] * 2)(idx)

        return vmap_index(), "aten::mul"

    @pytest.mark.precommit_torch_export
    def test_vmap(self, ie_device, precision, ir_version):
        if not self.use_torch_export():
            pytest.skip("torch.jit.trace cannot trace tensor indexing inside vmap")
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestDiffExport(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(3, 5).astype("float32"),)

    def create_model(self):
        class aten_diff(torch.nn.Module):
            def forward(self, x):
                return torch.diff(x, prepend=x[:, :1])

        return aten_diff(), "aten::diff"

    @pytest.mark.precommit_torch_export
    def test_diff(self, ie_device, precision, ir_version):
        if not self.use_torch_export():
            pytest.skip("aten::diff has no TorchScript converter")
        self._test(*self.create_model(), ie_device, precision, ir_version)
