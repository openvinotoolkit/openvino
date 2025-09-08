# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestTriuTril(PytorchLayerTest):
    def _prepare_input(self, shape, dtype):
        import numpy as np
        return (np.random.randn(*shape).astype(dtype),)

    def create_model(self, op, diagonal):

        import torch

        op_map = {
            "tril": torch.tril,
            "triu": torch.triu
        }

        pt_op = op_map[op]

        class aten_trilu(torch.nn.Module):
            def __init__(self, op, diagonal):
                super(aten_trilu, self).__init__()
                self.op = op
                self.diagonal = diagonal

            def forward(self, x):
                return self.op(x, self.diagonal)

        ref_net = None

        return aten_trilu(pt_op, diagonal), ref_net, f"aten::{op}"

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "int8", "uint8", "bool"])
    @pytest.mark.parametrize("diagonal", [0, 1, 2, -1, -2])
    @pytest.mark.parametrize("op", ["triu", "tril"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_trilu(self, dtype, diagonal, op, ie_device, precision, ir_version):
        self._test(*self.create_model(op, diagonal), ie_device, precision, ir_version, 
        kwargs_to_prepare_input={"shape": (4, 6), "dtype": dtype})


class TestTriuTrilTensor(PytorchLayerTest):
    def _prepare_input(self, shape, dtype):
        import numpy as np
        return (np.random.randn(*shape).astype(dtype),)

    def create_model(self, op, diagonal):

        import torch

        class aten_trilu(torch.nn.Module):
            def __init__(self, op, diagonal):
                super(aten_trilu, self).__init__()
                op_map = {
                    "tril": self.tril,
                    "tril_": self.tril_,
                    "triu": self.triu,
                    "triu_": self.triu_
                }
                self.diagonal = diagonal
                self.forward = op_map[op]

            def tril(self, x):
                return x.tril(self.diagonal), x

            def tril_(self, x):
                return x.tril_(self.diagonal), x

            def triu(self, x):
                return x.triu(self.diagonal), x

            def triu_(self, x):
                return x.triu_(self.diagonal), x

        ref_net = None

        return aten_trilu(op, diagonal), ref_net, f"aten::{op}"

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "int8", "uint8", "bool"])
    @pytest.mark.parametrize("diagonal", [0, 1, 2, -1, -2])
    @pytest.mark.parametrize("op", ["triu", "tril", "triu_", "tril_"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_trilu(self, dtype, diagonal, op, ie_device, precision, ir_version):
        self._test(*self.create_model(op, diagonal), ie_device, precision, ir_version, 
        kwargs_to_prepare_input={"shape": (4, 6), "dtype": dtype})
