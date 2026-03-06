# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestIsNested(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 3).astype(np.float32),)

    def create_model(self):
        import torch

        class prim_is_nested(torch.nn.Module):

            def forward(self, x):
                # is_nested is always lowered to constant false in OpenVINO.
                nested = x.is_nested
                if nested:
                    return x + 1
                return x

        return prim_is_nested(), "prim::is_nested"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_is_nested(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)
