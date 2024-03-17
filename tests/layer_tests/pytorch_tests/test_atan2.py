# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0i

import pytest
import torch
import math
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class TestAtan2(PytorchLayerTest):
    def _prepare_input(self, y, x, ref_dtype=None):
        inputs = [np.array(y).astype(ref_dtype) - np.array(y).astype(ref_dtype), np.array(x).astype(ref_dtype) - np.array(x).astype(ref_dtype)]
        if ref_dtype:
            inputs.append(np.zeros(1).astype(ref_dtype))
        return inputs

    def create_model(self, dtype=None, use_out=False):
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "uint8": torch.uint8,
            "int8": torch.int8,
            }
    
        class aten_atan2_out(torch.nn.Module):
            def __init__(self, out) -> None:
                super().__init__()
                self.out = torch.empty(25, dtype=out)
    
            def forward(self, y, x):
                return torch.atan2(input = y, other = x, out=self.out)

        class aten_atan2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
    
            def forward(self, y, x):
                return torch.atan2(input = y, other = x)
        
        dtype = dtype_map.get(dtype)

        if out_use:
            model_class = aten_atan2_out(dtype)
        else:
            model_class = aten_atan2()
                
    
        ref_net = None
    
        return model_class, ref_net, "aten::atan2"
        
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64", "int8", "uin8"])
    @pytest.mark.parametrize(
        "start,end,steps", [(0, 1), (0, 0), (1, -5), (1, 10), (-1, -5), (-1, -5), (1.25, -5.5)]
    )
    @pytest.mark.parametrize("use_out", [False, True])
    def test_linspace_with_out(self, dtype, use_out, y, x, ie_device, precision, ir_version):
        self._test(
            *self.create_model(dtype=dtype, use_out=use_out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"y": y, "x": x}
        )
