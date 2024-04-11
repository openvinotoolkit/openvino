# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSize(PytorchLayerTest):
    def _prepare_input(self, input_shape):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self):
        import torch

        class aten_size(torch.nn.Module):
            def forward(self, x):
                return torch.tensor(x.shape)

        ref_net = None

        op = aten_size()

        return op, ref_net, "aten::size"
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [[1,], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]])
    def test_size(self, input_shape, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, kwargs_to_prepare_input={"input_shape": input_shape})
