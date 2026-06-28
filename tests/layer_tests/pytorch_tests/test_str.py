# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestStr(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(self.value, dtype=np.int64),)

    def create_model(self):
        import torch

        class aten_str(torch.nn.Module):
            def forward(self, x):
                # str() of a scalar, consumed by len() -> exercises the
                # constant len(str(scalar)) support path.
                return torch.tensor(len(str(int(x))))

        return aten_str(), "aten::str"

    @pytest.mark.nightly
    @pytest.mark.parametrize("value", [0, 7, 42, 12345])
    def test_str(self, value, ie_device, precision, ir_version):
        self.value = value
        self._test(*self.create_model(), ie_device, precision, ir_version)
