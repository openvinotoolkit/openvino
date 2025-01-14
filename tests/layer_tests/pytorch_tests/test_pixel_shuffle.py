# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestPixelShuffle(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(*self.shape).astype(np.float32),)

    def create_model(self, upscale_factor):
        import torch
        import torch.nn.functional as F

        class aten_pixel_shuffle(torch.nn.Module):
            def __init__(self, upscale_factor):
                super(aten_pixel_shuffle, self).__init__()
                self.upscale_factor = upscale_factor

            def forward(self, x):
                return F.pixel_shuffle(x, self.upscale_factor)

        return aten_pixel_shuffle(upscale_factor), None, "aten::pixel_shuffle"

    @pytest.mark.parametrize(("upscale_factor,shape"), [(3, [1, 9, 4, 4]),
                                                        (2, [1, 2, 3, 8, 4, 4]),])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_pixel_shuffle(self, upscale_factor, shape, ie_device, precision, ir_version):
        self.shape = shape
        self._test(*self.create_model(upscale_factor),
                   ie_device, precision, ir_version)


class TestPixelUnshuffle(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(*self.shape).astype(np.float32),)

    def create_model(self, upscale_factor):
        import torch
        import torch.nn.functional as F

        class aten_pixel_unshuffle(torch.nn.Module):
            def __init__(self, upscale_factor):
                super(aten_pixel_unshuffle, self).__init__()
                self.upscale_factor = upscale_factor

            def forward(self, x):
                return F.pixel_unshuffle(x, self.upscale_factor)

        return aten_pixel_unshuffle(upscale_factor), None, "aten::pixel_unshuffle"

    @pytest.mark.parametrize(("upscale_factor,shape"), [(3, [1, 1, 12, 12]),
                                                        (2, [1, 2, 3, 2, 8, 8]),])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_pixel_unshuffle(self, upscale_factor, shape, ie_device, precision, ir_version):
        self.shape = shape
        self._test(*self.create_model(upscale_factor),
                   ie_device, precision, ir_version)

 
class TestChannelShuffle(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(*self.shape).astype(np.float32),)

    def create_model(self, groups):
        import torch
        import torch.nn.functional as F

        class aten_channel_shuffle(torch.nn.Module):
            def __init__(self, upscale_factor):
                super(aten_channel_shuffle, self).__init__()
                self.upscale_factor = upscale_factor

            def forward(self, x):
                return F.channel_shuffle(x, self.upscale_factor)

        return aten_channel_shuffle(groups), None, "aten::channel_shuffle"

    @pytest.mark.parametrize(("groups,shape"), [
        (3, [1, 9, 4, 4]),
        (2, [1, 8, 8, 4, 4]),
        (4, [4, 4, 2]),
        (5, [4, 10, 2, 10, 1, 1]),
        (1, [2, 3, 4])
        ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_channel_shuffle(self, groups, shape, ie_device, precision, ir_version):
        self.shape = shape
        self._test(*self.create_model(groups),
                   ie_device, precision, ir_version)
