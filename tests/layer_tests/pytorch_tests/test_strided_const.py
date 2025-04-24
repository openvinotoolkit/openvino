# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestStrides(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        x = np.random.randint(0, 10, [1, 3, 2, 2]).astype(np.float32)
        return (x,)

    def create_model(self):
        import torch

        class strided_const(torch.nn.Module):
            def __init__(self):
                super(strided_const, self).__init__()
                self.const = torch.randint(0, 10, [1, 3, 2, 2], dtype=torch.float32)
                self.const = self.const.to(memory_format=torch.channels_last)

            def forward(self, x):
                return x + self.const

        ref_net = None

        return strided_const(), ref_net, None

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_strides(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True)
