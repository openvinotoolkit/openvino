# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestAddMM(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2, 2), matrix1_shape=(2, 2), matrix2_shape=(2, 2)):
        import numpy as np
        return (
            np.random.randn(*input_shape).astype(np.float32),
            np.random.randn(*matrix1_shape).astype(np.float32),
            np.random.randn(*matrix2_shape).astype(np.float32)
        )

    def create_model(self, alpha, beta):
        import torch

        class aten_addmm(torch.nn.Module):
            def __init__(self, alpha, beta):
                super(aten_addmm, self).__init__()
                self.alpha = alpha
                self.beta = beta

            def forward(self, m0, m1, m2):
                return torch.addmm(m0, m1, m2, alpha=self.alpha, beta=self.beta)

        ref_net = None

        return aten_addmm(alpha, beta), ref_net, 'aten::addmm'

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {"input_shape": (3, 3), 'matrix1_shape': (3, 3), 'matrix2_shape': (3, 3)},
        {"input_shape": (2, 2), 'matrix1_shape': (2, 3), 'matrix2_shape': (3, 2)},
        {"input_shape": (10, 1), 'matrix1_shape': (10, 5), 'matrix2_shape': (5, 1)},
        {"input_shape": (1, 2), 'matrix1_shape': (1, 10), 'matrix2_shape': (10, 2)},
        {"input_shape": (1, 1), 'matrix1_shape': (1, 10), 'matrix2_shape': (10, 1)},

    ])
    @pytest.mark.parametrize("alpha,beta",
                             [(1., 1.), (0., 1.), (1., 0.), (1., 2.), (2., 1.), (-5., -6.), (3., 4.), (0.5, 0.75),
                              (1, 1)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_addmm(self, kwargs_to_prepare_input, alpha, beta, ie_device, precision, ir_version):
        self._test(*self.create_model(alpha, beta), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input)


class TestBAddBMM(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2, 2), matrix1_shape=(2, 2), matrix2_shape=(2, 2)):
        import numpy as np
        return (
            np.random.randn(*input_shape).astype(np.float32),
            np.random.randn(*matrix1_shape).astype(np.float32),
            np.random.randn(*matrix2_shape).astype(np.float32)
        )

    def create_model(self, alpha, beta):
        import torch

        class aten_addmm(torch.nn.Module):
            def __init__(self, alpha, beta):
                super(aten_addmm, self).__init__()
                self.alpha = alpha
                self.beta = beta

            def forward(self, m0, m1, m2):
                return torch.baddbmm(m0, m1, m2, alpha=self.alpha, beta=self.beta)

        ref_net = None

        return aten_addmm(alpha, beta), ref_net, 'aten::baddbmm'

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {"input_shape": (2, 3, 3), 'matrix1_shape': (2, 3, 3), 'matrix2_shape': (2, 3, 3)},
        {"input_shape": (2, 2, 2), 'matrix1_shape': (2, 2, 3), 'matrix2_shape': (2, 3, 2)},
        {"input_shape": (1, 10, 1), 'matrix1_shape': (1, 10, 5), 'matrix2_shape': (1, 5, 1)},
        {"input_shape": (5, 1, 2), 'matrix1_shape': (5, 1, 10), 'matrix2_shape': (5, 10, 2)},
        {"input_shape": (1, 1, 1), 'matrix1_shape': (1, 1, 10), 'matrix2_shape': (1, 10, 1)},

    ])
    @pytest.mark.parametrize("alpha,beta",
                             [  # beta==0 in some cases produce nan in pytorch
                                 (1., 1.),
                                 (0., 1.),
                                 (-5., -6.),
                                 (3., 4.),
                                 (0.5, 0.75),
                                 (1, 1)
                             ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_baddbmm(self, kwargs_to_prepare_input, alpha, beta, ie_device, precision, ir_version):
        self._test(*self.create_model(alpha, beta), ie_device, precision, ir_version,
                   kwargs_to_prepare_input=kwargs_to_prepare_input)
