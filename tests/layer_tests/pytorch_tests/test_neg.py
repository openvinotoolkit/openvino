# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class neg_model(torch.nn.Module):
    def forward(self, x):
        return x * (-x.shape[-1])


class TestNeg(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 4, 224, 224).astype(np.float32),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_neg(self, ie_device, precision, ir_version):
        self._test(neg_model(), None, "aten::neg",
                   ie_device, precision, ir_version,
                   dynamic_shapes_for_export={
                       "x": {3: torch.export.Dim("width")}
        })
