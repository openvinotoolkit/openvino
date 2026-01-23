# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_percent_format_string(torch.nn.Module):
    def forward(self):
        return "Hello %s" % ("World",)


class aten_percent_format_int(torch.nn.Module):
    def forward(self):
        return "Value: %d" % (42,)


class aten_percent_format_float(torch.nn.Module):
    def forward(self):
        return "Pi: %.2f" % (3.14159,)


class aten_percent_format_multiple(torch.nn.Module):
    def forward(self):
        return "Name: %s, Age: %d, Score: %.1f" % ("Alice", 30, 95.5)


class aten_percent_format_int_i(torch.nn.Module):
    def forward(self):
        return "Count: %i" % (100,)


class aten_percent_format_float_precision(torch.nn.Module):
    def __init__(self, precision):
        super().__init__()
        self.format_str = f"Value: %.{precision}f"
    
    def forward(self):
        return self.format_str % (3.14159265,)


class aten_percent_format_escaped(torch.nn.Module):
    def forward(self):
        return "Progress: %d%%" % (75,)


class TestPercentFormat(PytorchLayerTest):
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_percent_format_string(self, ie_device, precision, ir_version):
        self._test(aten_percent_format_string(), None, "aten::percentFormat",
                   ie_device, precision, ir_version)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_percent_format_int(self, ie_device, precision, ir_version):
        self._test(aten_percent_format_int(), None, "aten::percentFormat",
                   ie_device, precision, ir_version)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_percent_format_float(self, ie_device, precision, ir_version):
        self._test(aten_percent_format_float(), None, "aten::percentFormat",
                   ie_device, precision, ir_version)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_percent_format_multiple(self, ie_device, precision, ir_version):
        self._test(aten_percent_format_multiple(), None, "aten::percentFormat",
                   ie_device, precision, ir_version)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_percent_format_int_i(self, ie_device, precision, ir_version):
        self._test(aten_percent_format_int_i(), None, "aten::percentFormat",
                   ie_device, precision, ir_version)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("precision", [0, 1, 2, 3, 5])
    def test_percent_format_float_precision(self, precision, ie_device, ir_version):
        self._test(aten_percent_format_float_precision(precision), None, "aten::percentFormat",
                   ie_device, "FP32", ir_version)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_percent_format_escaped(self, ie_device, precision, ir_version):
        self._test(aten_percent_format_escaped(), None, "aten::percentFormat",
                   ie_device, precision, ir_version)
