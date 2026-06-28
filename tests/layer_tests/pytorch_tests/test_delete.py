# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestDelete(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(4, 3).astype(np.float32),)

    def create_model(self, idx):
        import torch

        class aten_delete(torch.nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                # build a list (tensor stacked along dim 0), delete one element,
                # then reduce the remaining list so the result depends on removal.
                lst = [x[0], x[1], x[2], x[3]]
                del lst[self.idx]
                out = lst[0]
                for i in range(1, len(lst)):
                    out = out + lst[i]
                return out

        return aten_delete(idx), "aten::Delete"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("idx", [0, 1, 2, 3, -1])
    def test_delete(self, idx, ie_device, precision, ir_version):
        self._test(*self.create_model(idx), ie_device, precision, ir_version)
