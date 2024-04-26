# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0i

import pytest
import torch
import math
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class TestAtan2(PytorchLayerTest):
    def _prepare_input(self, y, x, dtype1=None, dtype2=None):
        inputs = [np.array(y).astype(dtype1), np.array(x).astype(dtype2)]
        return inputs

    def create_model(self, dtype1=None, dtype2=None, use_out=False):
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "int16": torch.int16,
            "uint8":     torch.uint8,
            "int8": torch.int8,
            }
    
        class aten_atan2_out(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super().__init__()
                self.out = torch.empty(25, dtype=dtype)
    
            def forward(self, y, x):
                return torch.atan2(input = y, other = x, out=self.out)

        class aten_atan2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
    
            def forward(self, y, x):
                return torch.atan2(input = y, other = x)
        
        dtype1 = dtype_map.get(dtype1)
        dtype2 = dtype_map.get(dtype2)

        if use_out:
            model_class = aten_atan2_out(dtype1)
        else:
            model_class = aten_atan2()
                
        ref_net = None
    
        return model_class, ref_net, "aten::atan2"
        
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype1, dtype2", [(None, None), ("float32", "int32"), ("float64", "float64"), ("int32", "float64"), ("int64", "int16"), ("int8", "int8"), ("uint8", "uint8")])
    @pytest.mark.parametrize(
        "y, x", [(0, 1.5), (0, 0), (1.25, -5), (1, 10), (-1, -5.5), (-1, -5), (1.25, -5.5), (1.9, 2.9), [10, 9.9]]
    )
    @pytest.mark.parametrize("use_out", [False, True])
    def test_atan2_with_out(self, dtype1, dtype2, use_out, y, x, ie_device, precision, ir_version):
        self._test(
            *self.create_model(dtype2=dtype2, dtype1=dtype1, use_out=use_out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"y": y, "x": x}
        )
