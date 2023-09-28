# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestOneHot(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(*self.shape).astype(np.float32),)

    def create_model(self, upscale_factor):
        import torch
        import torch.nn.functional as F

        class aten_one_hot(torch.nn.Module):
            def __init__(self, upscale_factor):
                super(aten_one_hot, self).__init__()
                self.upscale_factor = upscale_factor

            def forward(self, x):
                return F.pixel_shuffle(x, self.upscale_factor)

        return aten_one_hot(upscale_factor), None, "aten::pixel_shuffle"

    @pytest.mark.parametrize(("upscale_factor,shape"), [(3, [1, 9, 4, 4]),
                                                        (2, [1, 2, 3, 8, 4, 4]),])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_one_hot(self, upscale_factor, shape, ie_device, precision, ir_version):
        self.shape = shape
        self._test(*self.create_model(upscale_factor),
                   ie_device, precision, ir_version)
