# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSparseMM(PytorchLayerTest):
    def _prepare_input(self, m, k, n):
        return (
            self.random.randn(m, k).astype(np.float32),
            self.random.randn(k, n).astype(np.float32),
        )

    def create_model(self):
        import torch

        class aten_sparse_mm(torch.nn.Module):
            def forward(self, a, b):
                # torch.sparse.mm dispatches to aten::_sparse_mm; a.to_sparse()
                # produces the sparse operand. Densely this equals a @ b.
                return torch.sparse.mm(a.to_sparse(), b)

        return aten_sparse_mm(), "aten::_sparse_mm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("m", "k", "n"), [(2, 3, 4), (5, 10, 7), (3, 5, 4)])
    def test_sparse_mm(self, m, k, n, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"m": m, "k": k, "n": n})
