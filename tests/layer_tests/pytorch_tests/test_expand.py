# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestExpand(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3).astype(np.float32),)

    def create_model(self, dim):
        import torch

        class aten_expand(torch.nn.Module):
            def __init__(self, dims):
                super(aten_expand, self).__init__()
                self.dims = dims

            def forward(self, x):
                return x.expand(self.dims)

        ref_net = None

        return aten_expand(dim), ref_net, "aten::expand"

    @pytest.mark.parametrize("dims", [(4, 3), (-1, -1), (1, 2, 3), (1, 2, 2, 3)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_expand(self, dims, ie_device, precision, ir_version):
        self._test(*self.create_model(dims), ie_device, precision, ir_version)


class TestExpandAs(PytorchLayerTest):
    def _prepare_input(self, input_shape, broadcast_shape):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32), np.random.randn(*broadcast_shape).astype(np.float32),)

    def create_model(self):
        import torch

        class aten_expand_as(torch.nn.Module):
            def __init__(self):
                super(aten_expand_as, self).__init__()

            def forward(self, x, y):
                return x.expand_as(y)

        ref_net = None

        return aten_expand_as(), ref_net, "aten::expand_as"

    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {'input_shape': [1, 2], "broadcast_shape": [1, 2]},
        {'input_shape': [1, 2], "broadcast_shape": [1, 4, 2]},
        {'input_shape': [1, 2], "broadcast_shape": [2, 2]},
        {'input_shape': [1, 2], "broadcast_shape": [2, 2, 2]},
        {'input_shape': [1, 2], "broadcast_shape": [1, 4, 2]},
        {'input_shape': [1, 2, 3], "broadcast_shape": [1, 2, 3]},
        {'input_shape': [1, 2, 3], "broadcast_shape": [1, 4, 2, 3]},
        {'input_shape': [1, 2, 3, 4], "broadcast_shape": [1, 2, 3, 4]},
        {'input_shape': [1, 2, 3, 4], "broadcast_shape": [1, 4, 2, 3, 4]},
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_expand(self, ie_device, precision, ir_version, kwargs_to_prepare_input):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, kwargs_to_prepare_input=kwargs_to_prepare_input)
