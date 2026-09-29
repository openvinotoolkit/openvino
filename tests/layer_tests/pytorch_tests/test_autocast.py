# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestWrapWithAutocastFX(PytorchLayerTest):
    """torch.autocast regions are exported as wrap_with_autocast higher-order op."""

    def _prepare_input(self):
        return (self.random.randn(2, 3, 8, 4).astype("float32"),
                self.random.randn(2, 3, 8, 5).astype("float32"))

    def create_model(self, multi_output, with_linear):
        import torch

        class AutocastRegion(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(5, 5)

            def forward(self, q, v):
                v = torch.nn.functional.pad(v, (0, 1), value=1.0)
                with torch.autocast(device_type="cpu", enabled=False):
                    kv = q.float().transpose(-1, -2) @ v.float()
                    out = q.float() @ kv
                    out = out[..., :-1] / (out[..., -1:] + 1e-5)
                    if with_linear:
                        out = self.proj(out)
                    if multi_output:
                        return out, kv
                    return out

        return AutocastRegion(), "wrap_with_autocast"

    @pytest.mark.parametrize("multi_output", [False, True])
    @pytest.mark.parametrize("with_linear", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_wrap_with_autocast(self, multi_output, with_linear, ie_device, precision, ir_version):
        self._test(*self.create_model(multi_output, with_linear), ie_device, precision, ir_version,
                   use_convert_model=True, fx_kind="wrap_with_autocast")
