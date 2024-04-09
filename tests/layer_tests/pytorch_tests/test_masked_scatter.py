# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging.version import parse as parse_version

from pytorch_layer_test_class import PytorchLayerTest


class TestMaskedScatter(PytorchLayerTest):
    def _prepare_input(self, shape, x_dtype="float32", mask_dtype="bool", out=False):
        import numpy as np
        x = np.random.randn(*shape).astype(x_dtype)
        mask = (x > 0.5).astype(mask_dtype)
        source = np.arange(np.size(x)).reshape(shape).astype(x_dtype)
        if not out:
            return (x, mask, source)
        y = np.zeros_like(x).astype(x_dtype)
        return (x, mask, source, y)

    def create_model(self, out=False, inplace=False):
        import torch

        class aten_masked_scatter(torch.nn.Module):
            def __init__(self, out, inplace):
                super(aten_masked_scatter, self).__init__()
                if inplace:
                    self.forward = self.forward_inplace
                if out:
                    self.forward = self.forward_out

            def forward(self, x, mask, source):
                return torch.masked_scatter(x, mask, source)
            
            def forward_out(self, x, mask, source, out):
                return torch.masked_scatter(x, mask, source, out=out), out

            def forward_inplace(self, x, mask, source):
                return x.masked_scatter_(mask, source), x

        ref_net = None

        return aten_masked_scatter(out, inplace), ref_net, "aten::masked_scatter" if not inplace else "aten::masked_scatter_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("shape", [[2, 5], [10, 10], [2, 3, 4], [10, 5, 10, 3], [2, 6, 4, 1]])
    @pytest.mark.parametrize("input_dtype", ["float32", "int32", "float", "int", "uint8"])
    @pytest.mark.parametrize("mask_dtype", ["bool"])
    @pytest.mark.parametrize("out", [True, False])
    def test_masked_scatter(self, shape, input_dtype, mask_dtype, out, ie_device, precision, ir_version):
        self._test(*self.create_model(out), ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"shape": shape, "x_dtype": input_dtype, "mask_dtype": mask_dtype, "out": out})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("shape", [[2, 5], [10, 10], [2, 3, 4], [10, 5, 10, 3], [2, 6, 4, 1]])
    @pytest.mark.parametrize("input_dtype", ["float32", "int32", "float", "int", "uint8"])
    @pytest.mark.parametrize("mask_dtype", ["bool"])
    def test_masked_scatter_inplace(self, shape, input_dtype, mask_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(inplace=True), ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"shape": shape, "x_dtype": input_dtype, "mask_dtype": mask_dtype})

    @pytest.mark.skipif(parse_version(torch.__version__) >= parse_version("2.1.0"), reason="pytorch 2.1 and above does not support nonboolean mask")
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("shape", [[2, 5], [10, 10], [2, 3, 4], [10, 5, 10, 3], [2, 6, 4, 1]])
    @pytest.mark.parametrize("input_dtype", ["float32", "int32", "float", "int", "uint8"])
    @pytest.mark.parametrize("mask_dtype", ["uint8"])
    @pytest.mark.parametrize("out", [True, False])
    def test_masked_scatter_u8(self, shape, input_dtype, mask_dtype, out, ie_device, precision, ir_version):
        self._test(*self.create_model(out), ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"shape": shape, "x_dtype": input_dtype, "mask_dtype": mask_dtype, "out": out})
        
    @pytest.mark.skipif(parse_version(torch.__version__) >= parse_version("2.1.0"), reason="pytorch 2.1 and above does not support nonboolean mask")
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("shape", [[2, 5], [10, 10], [2, 3, 4], [10, 5, 10, 3], [2, 6, 4, 1]])
    @pytest.mark.parametrize("input_dtype", ["float32", "int32", "float", "int", "uint8"])
    @pytest.mark.parametrize("mask_dtype", ["uint8"])
    def test_masked_scatter_inplace_u8(self, shape, input_dtype, mask_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(inplace=True), ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"shape": shape, "x_dtype": input_dtype, "mask_dtype": mask_dtype})
