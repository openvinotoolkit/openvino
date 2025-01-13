# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestNumel(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2)):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self):
        import torch
        class aten_numel(torch.nn.Module):

            def forward(self, x):
                return torch.numel(x)

        ref_net = None

        return aten_numel(), ref_net, 'aten::numel'

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {'input_shape': (1,)},
        {'input_shape': (2,)},
        {'input_shape': (2, 3)},
        {'input_shape': (3, 4, 5)},
        {'input_shape': (1, 2, 3, 4)},
        {'input_shape': (1, 2, 3, 4, 5)}
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_numel(self, kwargs_to_prepare_input, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input)
