# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestPermute(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, order, complex_type):
        import torch

        class aten_permute(torch.nn.Module):
            def __init__(self, order, complex_type):
                super().__init__()
                self.order = order
                self.complex_type = complex_type

            def forward(self, x):
                if self.complex_type:
                    x = torch.reshape(x, x.shape[:-1] + (-1, 2))
                    x = torch.view_as_complex(x)
                res = torch.permute(x, self.order)
                if self.complex_type:
                    res = torch.view_as_real(res)
                return res

        return aten_permute(order, complex_type), None, "aten::permute"

    @pytest.mark.parametrize("order", [[0, 2, 3, 1],
                                       [0, 3, 1, 2],
                                       [0, -1, 1, -2]])
    @pytest.mark.parametrize("complex_type", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_permute(self, order, complex_type, ie_device, precision, ir_version):
        self._test(*self.create_model(order, complex_type), ie_device, precision, ir_version)

class TestPermuteCopy(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, order):
        import torch

        class aten_permute_copy(torch.nn.Module):
            def __init__(self, order):
                super(aten_permute_copy, self).__init__()
                self.order = order

            def forward(self, x):
                return torch.permute_copy(x, self.order)

        ref_net = None

        return aten_permute_copy(order), ref_net, "aten::permute_copy"

    @pytest.mark.parametrize("order", [[0, 2, 3, 1], [0, 3, 1, 2], [0, -1, 1, -2]])
    @pytest.mark.precommit_fx_backend
    def test_permute_copy(self, order, ie_device, precision, ir_version):
        self._test(*self.create_model(order), ie_device, precision, ir_version)


class TestPermuteList(PytorchLayerTest):
    def _prepare_input(self, permute_shape):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),
                np.random.randn(*permute_shape).astype(np.float32))

    def create_model(self):
        import torch

        class aten_permute_list(torch.nn.Module):
            def forward(self, x, y):
                y_shape = y.shape
                return torch.permute(x, [y_shape[0] - 1, y_shape[1] - 1, y_shape[2] - 1, y_shape[3] - 1])

        ref_net = None

        return aten_permute_list(), ref_net, ["aten::permute", "prim::ListConstruct"]

    @pytest.mark.parametrize("order", [[1, 3, 4, 2], [1, 4, 2, 3]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_permute_list(self, order, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"permute_shape": order},
                   dynamic_shapes=ie_device != "GPU")
