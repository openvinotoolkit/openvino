# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import random

from pytorch_layer_test_class import PytorchLayerTest


class TestExpand(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3).astype(np.float32),)

    def create_model(self, dim, op_type="expand"):
        import torch

        class aten_expand(torch.nn.Module):
            def __init__(self, dims, op_type="expand"):
                super(aten_expand, self).__init__()
                self.dims = dims
                if op_type == "broadcast_to":
                    self.forward = self.forward_broadcast

            def forward(self, x):
                return x.expand(self.dims)

            def forward_broadcast(self, x):
                return x.broadcast_to(self.dims)

        ref_net = None

        return aten_expand(dim, op_type), ref_net, f"aten::{op_type}"

    @pytest.mark.parametrize("dims", [(4, 3), (-1, -1), (1, 2, 3), (1, 2, 2, 3)])
    @pytest.mark.parametrize("op_type", ["expand", "broadcast_to"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_expand(self, dims, op_type, ie_device, precision, ir_version):
        self._test(*self.create_model(dims, op_type), ie_device, precision, ir_version)

class TestExpandList(PytorchLayerTest):
    def _prepare_input(self, broadcast_shape):
        import numpy as np
        return (np.random.randn(1, 3).astype(np.float32), np.random.randn(*broadcast_shape).astype(np.float32))

    def create_model(self, op_type="expand"):
        import torch

        class aten_expand(torch.nn.Module):
            def __init__(self, op_type="expand"):
                super(aten_expand, self).__init__()
                if op_type == "broadcast_to":
                    self.forward = self.forward_broadcast

            def forward(self, x, y):
                y_shape = y.shape
                return x.expand([y_shape[0], y_shape[1]])

            def forward_broadcast(self, x, y):
                y_shape = y.shape
                return x.broadcast_to([y_shape[0], y_shape[1]])

        ref_net = None

        return aten_expand(op_type), ref_net, [f"aten::{op_type}", "prim::ListConstruct"]

    @pytest.mark.parametrize("dims", [(3, 3), (2, 3), (1, 3), [4, 3]])
    @pytest.mark.parametrize("op_type", ["expand", "broadcast_to"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_expand(self, dims, op_type, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type), ie_device, precision, ir_version, kwargs_to_prepare_input={"broadcast_shape": dims})


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
    @pytest.mark.precommit_torch_export
    def test_expand(self, ie_device, precision, ir_version, kwargs_to_prepare_input):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, kwargs_to_prepare_input=kwargs_to_prepare_input)

class TestDynamicExpand(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        last_dym = random.randint(2,8)
        return (np.random.randn(1, 3, 1).astype(np.float32), last_dym)

    def create_model(self, dim):
        import torch

        class aten_expand(torch.nn.Module):
            def __init__(self, dims):
                super(aten_expand, self).__init__()
                self.dims = dims

            # TODO: Remove the add op after fixing the issue with expand being the last node
            def forward(self, x, dym):
                return torch.add(x.expand((self.dims+(dym,))), 0)

        ref_net = None

        return aten_expand(dim), ref_net, f"aten::expand"

    @pytest.mark.parametrize("dims", [(4, 3), (-1, -1)])
    @pytest.mark.precommit_fx_backend
    def test_dynamic_expand(self, dims, ie_device, precision, ir_version):
        self._test(*self.create_model(dims), ie_device, precision, ir_version)
