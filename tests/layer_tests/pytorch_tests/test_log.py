# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestLog(PytorchLayerTest):
    def _prepare_input(self, dtype):
        import numpy as np
        return (np.random.uniform(2, 16, (1, 10)).astype(dtype),)

    def create_model(self, op):
        import torch

        ops = {
            "log": torch.log,
            "log_": torch.log_,
            "log2": torch.log2,
            "log2_": torch.log2_,
            "log10": torch.log10,
            "log1p": torch.log1p,
            "log1p_": torch.log1p_
        }

        op_fn = ops[op]

        class aten_log(torch.nn.Module):
            def __init__(self, op):
                super(aten_log, self).__init__()
                self.op = op

            def forward(self, x):
                return self.op(x)

        ref_net = None

        return aten_log(op_fn), ref_net, f"aten::{op}"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("op", "input_dtype"),
                             [["log", "float32"], 
                             ["log", "int32"], 
                             ["log_", "float32"], 
                             ["log2", "float32"], 
                             ["log2", "int32"], 
                             ["log2_", "float32"],
                             ["log10", "float32"],
                             ["log1p", "float32"], 
                             ["log1p", "int32"], 
                             ["log1p_", "float32"]])
    def test_log(self, op, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(op), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"dtype": input_dtype})