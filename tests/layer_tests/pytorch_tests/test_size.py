# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSize(PytorchLayerTest):
    def _prepare_input(self, input_shape, complex_type):
        import numpy as np
        if complex_type:
            input_shape += [2]
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self, complex_type):
        import torch

        class aten_size(torch.nn.Module):
            def __init__(self, complex_type):
                super().__init__()
                self.complex_type = complex_type

            def forward(self, x):
                if self.complex_type:
                    x = torch.view_as_complex(x)
                return torch.tensor(x.shape)

        op = aten_size(complex_type)

        return op, None, "aten::size"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [[1,],
                                             [1, 2],
                                             [1, 2, 3],
                                             [1, 2, 3, 4],
                                             [1, 2, 3, 4, 5]])
    @pytest.mark.parametrize("complex_type", [True, False])
    def test_size(self, input_shape, complex_type, ie_device, precision, ir_version):
        self._test(*self.create_model(complex_type), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape,
                                            "complex_type": complex_type})
