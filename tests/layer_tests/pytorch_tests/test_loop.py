# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import platform

import pytest
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestLoopWithAlias(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(*self.shape).astype(np.float32),)

    def create_model(self, n):
        import torch

        class loop_alias_model(torch.nn.Module):
            def __init__(self, n):
                super(loop_alias_model, self).__init__()
                self.n = n

            def forward(self, x):
                N = x.shape[1]
                res = torch.zeros(1, self.n, dtype=torch.long)
                d = torch.ones(1, N) * 1e10
                f = torch.zeros(1, dtype=torch.long)

                for i in range(self.n):
                    res[:, i] = f
                    _d = torch.sum((x - x[0, f, :]) ** 2, -1)
                    m = _d < d
                    d[m] = _d[m]
                    f = torch.max(d, -1)[1]
                return res

        return loop_alias_model(n), None, ["prim::Loop", "aten::copy_"]

    @pytest.mark.parametrize("s,n", [([1, 1024, 3], 512), ([1, 512, 3], 128)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_loop_alias(self, s, n, ie_device, precision, ir_version):
        self.shape = s
        self._test(*self.create_model(n), ie_device, precision,
                   ir_version, use_convert_model=True)
