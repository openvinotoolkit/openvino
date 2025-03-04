# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSilu(PytorchLayerTest):
    def _prepare_input(self, inp_type="mixed", out=False):
        import numpy as np
        inp = np.arange(0, 10).astype(np.float32)
        if inp_type == "negative":
            inp[0] = 1
            inp = -1 * inp
        elif inp_type == "positive":
            inp[0] = 11
        elif inp_type == "zeros":
            inp *= 0
        else:
            idx = np.random.choice(inp, 3)
            inp[idx.astype(int)] *= -1
        if out:
            return (inp, np.zeros_like(inp))
        return (inp, )

    def create_model(self, out):
        import torch

        class aten_sign(torch.nn.Module):
            def __init__(self, out):
                super(aten_sign, self).__init__()
                if out:
                    self.forward = self.forward_out

            def forward(self, x):
                return torch.sign(x)

            def forward_out(self, x, out):
                return torch.sign(x), out

        ref_net = None

        return aten_sign(out), ref_net, "aten::sign"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("input_type", ["zeros", "positive", "negative", "mixed"])
    @pytest.mark.parametrize("out", [True, False])
    def test_sign(self, input_type, out, ie_device, precision, ir_version):
        self._test(*self.create_model(out), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"inp_type": input_type, "out": out})
