# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestLog(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randint(0, 255, (20, 30, 40, 50)),)

    def create_model(self):
        import torch

        class aten_or(torch.nn.Module):
            def forward(self, x):
                res = torch.ByteTensor(x.size()).zero_()
                res[:, :, :, 1:] = res[:, :, :, 1:] | (x[:, :, :, 1:] != x[:, :, :, :-1])
                res[:, :, :, :-1] = res[:, :, :, :-1] | (x[:, :, :, 1:] != x[:, :, :, :-1])
                return res.float()

        return aten_or(), None, "aten::__or__"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_or(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   dynamic_shapes=False, trace_model=True, use_convert_model=True)
