# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestBucketize(PytorchLayerTest):

    def _prepare_input(self, input_shape, boundaries_range, input_dtype, boundaries_dtype):
        return (
            np.random.randn(*input_shape).astype(input_dtype), 
            np.arange(*boundaries_range).astype(boundaries_dtype))

    def create_model(self, out_int32, right, is_out):
        class aten_bucketize(torch.nn.Module):

            def __init__(self, out_int32, right, is_out) -> None:
                super().__init__()
                self.out_int32 = out_int32
                self.right = right 
                self.is_out = is_out

            def forward(self, input, boundaries):
                if self.is_out:
                    output_dtype = torch.int32 if self.out_int32 else torch.int64
                    output = torch.zeros_like(input, dtype=output_dtype)
                    torch.bucketize(input, boundaries, out_int32=self.out_int32, right=self.right, out=output)
                    return output
                else:
                    return torch.bucketize(input, boundaries, out_int32=self.out_int32, right=self.right)

        ref_net = None

        return aten_bucketize(out_int32, right, is_out), ref_net, "aten::bucketize"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("out_int32", [True, False])
    @pytest.mark.parametrize("right", [True, False])
    @pytest.mark.parametrize("is_out", [True, False])
    @pytest.mark.parametrize("input_shape", [[1, ], [2, 1], [2, 2, 1]])
    @pytest.mark.parametrize("input_dtype", ["float32", "int32"])
    @pytest.mark.parametrize("boundaries_range", [[1, 10], (100, 200)])
    @pytest.mark.parametrize("boundaries_dtype", ["float32", "int32"])
    def test_bucketize(self, input_shape, boundaries_range, input_dtype, boundaries_dtype, out_int32, right, is_out, ie_device, precision, ir_version):
        self._test(*self.create_model(out_int32, right, is_out), ie_device, precision, ir_version, kwargs_to_prepare_input={
            "input_shape": input_shape, "input_dtype": input_dtype,
            "boundaries_range": boundaries_range, "boundaries_dtype": boundaries_dtype,
        })
