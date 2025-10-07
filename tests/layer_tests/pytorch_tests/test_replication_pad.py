# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestReplicationPad(PytorchLayerTest):
    def _prepare_input(self, n):
        import numpy as np
        return (np.random.randn(*(2, 5, 6, 7, 8)[:n+2]).astype(np.float32),)

    def create_model(self, pad, n):

        import torch

        class aten_replication_pad(torch.nn.Module):
            def __init__(self, pad=None):
                super().__init__()
                self.pad = pad

            def forward_1d(self, x):
                return torch.ops.aten.replication_pad1d(x, self.pad)

            def forward_2d(self, x):
                return torch.ops.aten.replication_pad2d(x, self.pad)

            def forward_3d(self, x):
                return torch.ops.aten.replication_pad3d(x, self.pad)

        model = aten_replication_pad(pad)
        model.forward = model.__getattribute__(f"forward_{n}d")
        return model, None, f"aten::replication_pad{n}d"

    @pytest.mark.parametrize(("pad", "n"), [
        ((1, 1), 1),
        ((0, 2), 1),
        ((3, 1), 1),
        ((0, 0), 1),
        ((1, 1, 1, 1), 2),
        ((0, 2, 0, 2), 2),
        ((3, 1, 5, 2), 2),
        ((0, 0, 0, 0), 2),
        ((1, 1, 1, 1, 1, 1), 3),
        ((0, 2, 0, 2, 0, 2), 3),
        ((3, 1, 5, 2, 4, 3), 3),
        ((0, 0, 0, 0, 0, 0), 3),
    ])
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_replication_pad(self, pad, n, ie_device, precision, ir_version):
        self._test(*self.create_model(pad, n),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"n": n})
