# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from typing import List

from pytorch_layer_test_class import PytorchLayerTest


class TestTolist(PytorchLayerTest):

    def _prepare_input(self):
        return (self.random.randn(3).astype(np.float32),)

    def create_model(self):
        import torch

        class tolist_model(torch.nn.Module):

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Annotate the result so TorchScript keeps prim::tolist
                # in the graph instead of raising a type-hint error.
                y: List[float] = x.tolist()
                # Use the list so the node is not dead-code-eliminated.
                if len(y) > 0:
                    return x
                return x + 1

        return tolist_model(), "prim::tolist"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_tolist(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)
