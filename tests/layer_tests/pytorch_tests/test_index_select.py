# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestIndexSelect(PytorchLayerTest):
    def _prepare_input(self, index, out=False, dim=0):
        import numpy as np
        index = np.array(index).astype(np.int32)
        input_data = np.random.randn(2, 3, 10, 10).astype(np.float32)
        if not out:
            return (input_data, index)
        out = np.zeros_like(np.take(input_data, axis=dim, indices=index))
        return (input_data, index, out)

    def create_model(self, dim, out=False):
        import torch

        class aten_index_select(torch.nn.Module):
            def __init__(self, dim, out=False):
                super(aten_index_select, self).__init__()
                self.dim = dim
                if out:
                    self.forward = self.forward_out

            def forward(self, x, indices):
                return torch.index_select(x, self.dim, indices)

            def forward_out(self, x, indices, out):
                return out, torch.index_select(x, self.dim, indices, out=out)

        ref_net = None

        return aten_index_select(dim, out), ref_net, "aten::index_select"

    @pytest.mark.parametrize("dim", [0, 1, 2, 3, -1, -2, -3])
    @pytest.mark.parametrize("indices", [[0, 1], [0], [1, 0]])
    @pytest.mark.parametrize("out", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_index_select(self, dim, out, indices, ie_device, precision, ir_version):
        self._test(*self.create_model(dim, out), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"index": indices, "out": out, "dim": dim})
