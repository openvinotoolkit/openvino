# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestInt(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.array([3.7], dtype=np.float32), )

    def create_model(self):
        import torch

        class aten_int(torch.nn.Module):
            def forward(self, x):
                return x.int()

        return aten_int(), "aten::to"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_int(self, ie_device, precision, ir_version):
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version
        )