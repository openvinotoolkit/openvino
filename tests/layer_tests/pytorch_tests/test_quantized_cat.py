# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_quantized_cat(torch.nn.Module):
    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = float(scale)
        self.zero_point = int(zero_point)

    def forward(self, x):
        x = torch.quantize_per_tensor(x, 1., 0, torch.quint8)
        return torch.dequantize(torch.ops.quantized.cat([x, x], 1, self.scale, self.zero_point))


class aten_append_quantized_cat(torch.nn.Module):
    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = float(scale)
        self.zero_point = int(zero_point)
    def forward(self, x):
        x = torch.quantize_per_tensor(x, 1., 0, torch.quint8)
        list = []
        list.append(x)
        list.append(x)
        return torch.dequantize(torch.ops.quantized.cat(list, 1, self.scale, self.zero_point))


class aten_loop_append_quantized_cat(torch.nn.Module):
    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = float(scale)
        self.zero_point = int(zero_point)
    def forward(self, x):
        x = torch.quantize_per_tensor(x, 1., 0, torch.quint8)
        list = []
        for i in range(3):
            list.append(x)
        return torch.dequantize(torch.ops.quantized.cat(list, 1, self.scale, self.zero_point))


class aten_add_quantized_cat(torch.nn.Module):
    def __init__(self, scale, zero_point):
        super().__init__()
        self.scale = float(scale)
        self.zero_point = int(zero_point)
    def forward(self, x):
        x = torch.quantize_per_tensor(x, 1., 0, torch.quint8)
        list = [x, x]
        list2 = list + [x, x]
        return torch.dequantize(torch.ops.quantized.cat(list2, 1, self.scale, self.zero_point))


class TestQuantizedCat(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 1, 3).astype(np.float32),)

    @pytest.mark.parametrize("scale", [1, 0.3, 1.3])
    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantized_cat(self, scale, zero_point, ie_device, precision, ir_version):
        self._test(aten_quantized_cat(scale, zero_point), None, ["quantized::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version, quantized_ops=True, freeze_model=False, quant_size=scale)

    @pytest.mark.parametrize("scale", [1, 0.3, 1.3])
    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_append_quantized_cat(self, scale, zero_point, ie_device, precision, ir_version):
        self._test(aten_append_quantized_cat(scale, zero_point), None, ["quantized::cat", "aten::append", "prim::ListConstruct"],
                   ie_device, precision, ir_version, quantized_ops=True, freeze_model=False, quant_size=scale)

    @pytest.mark.parametrize("scale", [1, 0.3, 1.3])
    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason="Transformation RemoveMultiSubGraphOpDanglingParamsResults doesn't support removing unused merged inputs, ticket 112833.")
    def test_loop_append_quantized_cat(self, scale, zero_point, ie_device, precision, ir_version):
        self._test(aten_loop_append_quantized_cat(scale, zero_point), None, ["quantized::cat", "aten::append", "prim::ListConstruct", "prim::Loop"],
                   ie_device, precision, ir_version, quantized_ops=True, freeze_model=False, quant_size=scale)

    @pytest.mark.parametrize("scale", [1, 0.3, 1.3])
    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_add_quantized_cat(self, scale, zero_point, ie_device, precision, ir_version):
        self._test(aten_add_quantized_cat(scale, zero_point), None, ["quantized::cat", "aten::add", "prim::ListConstruct"],
                   ie_device, precision, ir_version, trace_model=False, quantized_ops=True, freeze_model=False, quant_size=scale)
