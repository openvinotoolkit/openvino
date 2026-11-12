# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestPatchMerging(PytorchLayerTest):
    """Swin-style PatchMerging followed by a window block that reads H and W from its output.

    The merge is view -> four step-2 slices -> cat -> view. With dynamic input shapes the window
    block's `view(B, H, W, C)` is a ShapeOf sub-graph of the concatenated tensor (H/2, W/2) that
    the symbolic shape optimizations must not re-source to the tensor before the slices (H, W).
    """

    def _prepare_input(self, image_hw):
        return (self.random.randn(1, 3, *image_hw),)

    def create_model(self):
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
                b, c, h, w = x.shape
                x = x.flatten(2).transpose(1, 2)
                # PatchMerging
                x = x.view(b, h, w, c)
                x0 = x[:, 0::2, 0::2, :]
                x1 = x[:, 1::2, 0::2, :]
                x2 = x[:, 0::2, 1::2, :]
                x3 = x[:, 1::2, 1::2, :]
                x = torch.cat([x0, x1, x2, x3], -1)
                _, h, w, _ = x.shape
                x = x.view(b, -1, 4 * c)
                x = self.reduction(self.norm(x))
                # window block of the next stage
                b, _, c = x.shape
                shortcut = x
                x = self.norm1(x)
                x = x.view(b, h, w, c)
                ws = self.window_size
                pad_r = (ws - w % ws) % ws
                pad_b = (ws - h % ws) % ws
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
                _, hp, wp, _ = x.shape
                nh, nw = hp // ws, wp // ws
                windows = x.view(b, nh, ws, nw, ws, c).permute(0, 1, 3, 2, 4, 5).contiguous()
                windows = windows.view(-1, ws * ws, c)
                windows = self.mix(windows)
                x = windows.view(b, nh, nw, ws, ws, c).permute(0, 1, 3, 2, 4, 5).contiguous()
                x = x.view(b, hp, wp, c)
                if pad_r > 0 or pad_b > 0:
                    x = x[:, :h, :w, :].contiguous()
                x = x.view(b, h * w, c)
                x = shortcut + x
                return x.view(b, h, w, c).permute(0, 3, 1, 2)

        return patch_merging_block(), "aten::slice"

    # merged H x W is 8x12 (multiple of the window, no padding) and 5x7 (padded)
    @pytest.mark.parametrize("image_hw", [(64, 96), (40, 56)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_patch_merging(self, image_hw, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"image_hw": image_hw},
                   use_convert_model=True, trace_model=True)
