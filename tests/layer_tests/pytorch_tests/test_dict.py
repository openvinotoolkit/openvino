# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestDict(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 5, 3, 4).astype(np.float32),)

    def create_model(self):
        class aten_dict(torch.nn.Module):
            def forward(self, x):                
                return {"b": x, "a": x + x, "c": 2 * x}, x / 2

        return aten_dict(), None, "prim::DictConstruct"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_dict(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True)
