# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestPermute(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, order):
        import torch

        class aten_permute(torch.nn.Module):
            def __init__(self, order):
                super(aten_permute, self).__init__()
                self.order = order

            def forward(self, x):
                return torch.permute(x, self.order)

        ref_net = None

        return aten_permute(order), ref_net, "aten::permute"

    @pytest.mark.parametrize("order", [[0, 2, 3, 1], [0, 3, 1, 2]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_permute(self, order, ie_device, precision, ir_version):
        self._test(*self.create_model(order), ie_device, precision, ir_version)

class TestPermuteList(PytorchLayerTest):
    def _prepare_input(self, permute_shape):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32), np.random.randn(*permute_shape).astype(np.float32))

    def create_model(self):
        import torch

        class aten_permute(torch.nn.Module):

            def forward(self, x, y):
                y_shape = y.shape
                return torch.permute(x, [y_shape[0] - 1, y_shape[1] - 1, y_shape[2] - 1, y_shape[3] - 1])

        ref_net = None

        return aten_permute(), ref_net, ["aten::permute", "prim::ListConstruct"]

    @pytest.mark.parametrize("order", [[1, 3, 4, 2], [1, 4, 2, 3]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_permute(self, order, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"permute_shape": order}, dynamic_shapes=ie_device != "GPU")
