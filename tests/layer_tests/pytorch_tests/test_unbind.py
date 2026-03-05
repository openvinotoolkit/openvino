# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestUnbind(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.uniform(0, 50, (3, 3, 3, 3), dtype=np.float32),)

    def create_model(self, shape):
        import torch

        class aten_unbind(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                # Create aten::unbind -> ListUnpack
                a, b, c = torch.unbind(x, self.dim)
                return b


        return aten_unbind(shape), "aten::unbind"

    @pytest.mark.parametrize(("dim"), [0, 1, 2, 3])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_unbind(self, dim, ie_device, precision, ir_version):
        self._test(*self.create_model(dim), ie_device, precision, ir_version)
