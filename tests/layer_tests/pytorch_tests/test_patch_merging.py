# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestPatchMerging(PytorchLayerTest):
    """Swin-style PatchMerging (view -> four step-2 slices -> cat -> view) followed by a
    window-partition block that reads H and W from the concatenated tensor's shape.

    With dynamic input shapes the block's `view(B, H, W, C)` pattern is a ShapeOf sub-graph of the
    concatenated tensor (H/2, W/2), which must not be confused with the shape of the tensor before
    the strided slices (H, W) by the symbolic shape optimizations.
    """

    def _prepare_input(self):
        return (self.random.randn(1, 3, *self.image_hw),)

    def create_model(self, window_size):
        class patch_merging_block(torch.nn.Module):
            def __init__(self, in_channels=3, embed_dim=8, window_size=4):
                super().__init__()
                self.window_size = window_size
                self.proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
                self.norm = torch.nn.LayerNorm(4 * embed_dim)
                self.reduction = torch.nn.Linear(4 * embed_dim, 2 * embed_dim, bias=False)
                self.norm1 = torch.nn.LayerNorm(2 * embed_dim)
                self.mix = torch.nn.Linear(2 * embed_dim, 2 * embed_dim)

            def forward(self, image):
                x = self.proj(image)
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
                # PatchMerging
                x = x.view(B, H, W, C)
                x0 = x[:, 0::2, 0::2, :]
                x1 = x[:, 1::2, 0::2, :]
                x2 = x[:, 0::2, 1::2, :]
                x3 = x[:, 1::2, 1::2, :]
                x = torch.cat([x0, x1, x2, x3], -1)
                _, H, W, _ = x.shape
                x = x.view(B, -1, 4 * C)
                x = self.reduction(self.norm(x))
                # window block of the next stage
                B, L, C = x.shape
                shortcut = x
                x = self.norm1(x)
                x = x.view(B, H, W, C)
                ws = self.window_size
                pad_r = (ws - W % ws) % ws
                pad_b = (ws - H % ws) % ws
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
                _, Hp, Wp, _ = x.shape
                nH, nW = Hp // ws, Wp // ws
                windows = x.view(B, nH, ws, nW, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
                windows = windows.view(-1, ws * ws, C)
                windows = self.mix(windows)
                x = windows.view(B, nH, nW, ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
                x = x.view(B, Hp, Wp, C)
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :H, :W, :].contiguous()
                x = x.view(B, H * W, C)
                x = shortcut + x
                return x.view(B, H, W, C).permute(0, 3, 1, 2)

        return patch_merging_block(window_size=window_size), "aten::slice"

    @pytest.mark.parametrize("image_hw", [(64, 96), (40, 56)])
    @pytest.mark.parametrize("window_size", [4])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_patch_merging(self, image_hw, window_size, ie_device, precision, ir_version):
        self.image_hw = image_hw
        self._test(*self.create_model(window_size), ie_device, precision, ir_version,
                   use_convert_model=True, trace_model=True, dynamic_shapes=True)
