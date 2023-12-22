# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSum(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        # This test had sporadically failed by accuracy. Try to resolve that by using int numbers in input
        return (np.random.randint(-10, 10, (1, 3, 224, 224)).astype(np.float32),)

    def create_model(self, axes, keep_dims):

        import torch

        class aten_sum(torch.nn.Module):
            def __init__(self, axes=None, keep_dims=None):
                super(aten_sum, self).__init__()
                self.axes = axes
                self.keep_dims = keep_dims

            def forward(self, x):
                if self.axes is None and self.keep_dims is None:
                    return torch.sum(x)
                if self.axes is not None and self.keep_dims is None:
                    return torch.sum(x, self.axes)
                return torch.sum(x, self.axes, self.keep_dims)

        ref_net = None

        return aten_sum(axes, keep_dims), ref_net, "aten::sum"

    @pytest.mark.parametrize("axes,keep_dim",
                             [(None, None), (None, False), (-1, None), (1, None), ((2, 3), False), ((3, 2), True)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sum(self, axes, keep_dim, ie_device, precision, ir_version):
        self._test(*self.create_model(axes, keep_dim),
                   ie_device, precision, ir_version)


class TestSumBool(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randint(0, 2, (1, 3, 20, 24)).astype(bool),)

    def create_model(self, axes, keep_dims):

        import torch

        class aten_sum(torch.nn.Module):
            def __init__(self, axes=None, keep_dims=None):
                super(aten_sum, self).__init__()
                self.axes = axes
                self.keep_dims = keep_dims

            def forward(self, x):
                x = x.to(torch.bool)
                if self.axes is None and self.keep_dims is None:
                    return torch.sum(x)
                if self.axes is not None and self.keep_dims is None:
                    return torch.sum(x, self.axes)
                return torch.sum(x, self.axes, self.keep_dims)

        ref_net = None

        return aten_sum(axes, keep_dims), ref_net, "aten::sum"

    @pytest.mark.parametrize("axes,keep_dim",
                             [(None, None), (None, False), (-1, None), (1, None), ((2, 3), False), ((3, 2), True)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sum(self, axes, keep_dim, ie_device, precision, ir_version):
        self._test(*self.create_model(axes, keep_dim),
                   ie_device, precision, ir_version)