# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestResolveConjNeg(PytorchLayerTest):
    def _prepare_input(self, dtype="float32"):
        import numpy as np
        return (np.random.randn(2, 4).astype(dtype),)

    def _prepare_input_complex(self):
        import numpy as np
        return (np.array([[2+3j, 3-2j, 4-9j,10+1j], [1-3j, 3+2j, 4+9j,10-5j]]), )


    def create_model(self, op_type):
        import torch
        
        ops = {
            "resolve_conj": torch.resolve_conj,
            "resolve_neg": torch.resolve_neg
        }

        op = ops[op_type]

        class aten_resolve(torch.nn.Module):
            def __init__(self, op):
                super(aten_resolve, self).__init__()
                self.op = op

            def forward(self, x):
                return self.op(x)

        ref_net = None

        return aten_resolve(op), ref_net, f"aten::{op_type}"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("op_type", ["resolve_neg", "resolve_conj"])
    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    def test_reslove(self, op_type, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type), ie_device, precision, ir_version, kwargs_to_prepare_input={"dtype": dtype})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("op_type", ["resolve_neg", "resolve_conj"])
    @pytest.mark.xfail(reason="complex dtype is not supported yet")
    def test_resolve_complex(self, op_type, ie_device, precision, ir_version):
        self._prepare_input = self._prepare_input_complex
        self._test(*self.create_model(op_type), ie_device, precision, ir_version)
