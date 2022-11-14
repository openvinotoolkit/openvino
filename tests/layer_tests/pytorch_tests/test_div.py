# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestDiv(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.array([[-0.3711, -1.9353, -0.4605, -0.2917],
                          [0.1815, -1.0111,  0.9805, -1.5923],
                          [0.1062,  1.4581,  0.7759, -1.2344],
                          [-0.1830, -0.0313,  1.1908, -1.4757]], dtype=np.float32),)

    def create_model(self, rounding_mode):
        import torch

        class aten_div(torch.nn.Module):
            def __init__(self, rounding_mode):
                super(aten_div, self).__init__()
                self.y = torch.tensor([0.8032,  0.2930, -0.8113, -0.2308])
                self.rounding_mode = rounding_mode

            def forward(self, x):
                return torch.div(x, self.y, rounding_mode=self.rounding_mode)

        ref_net = None

        return aten_div(rounding_mode), ref_net, "aten::div"

    @pytest.mark.parametrize("rounding_mode", [None,
                                               'floor',
                                               pytest.param('trunc', marks=pytest.mark.xfail)]) # trunc is not supported
    @pytest.mark.nightly
    def test_exp(self, rounding_mode, ie_device, precision, ir_version):
        self._test(*self.create_model(rounding_mode),
                   ie_device, precision, ir_version)
