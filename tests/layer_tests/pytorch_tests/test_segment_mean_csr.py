# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from torch_scatter import segment_csr

from pytorch_layer_test_class import PytorchLayerTest

class torch_segment_mean_csr(torch.nn.Module):
    def forward(self, src, indptr):
        return segment_csr(src, indptr, reduce="mean")

class TestSegmentMeanCSR(PytorchLayerTest):
    def _prepare_input(self):
        src = np.random.randn(10, 6, 64).astype(np.float32)
        indptr = np.array([0, 2, 5, 6], dtype=np.int64).reshape(1, -1)
        return (src, indptr)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_segment_mean_csr(self, ie_device, precision, ir_version):
        model = torch_segment_mean_csr()
        self._test(model, None, "torch_scatter::segment_mean_csr", ie_device, 
                  precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_segment_mean_csr_shape_verification(self, ie_device, precision, ir_version):
        """Test that verifies the output shape is correct based on indptr size"""
        model = torch_segment_mean_csr()
        
        # Define various test cases with different indptr sizes
        test_cases = [
            # (src_shape, indptr_values, expected_out_shape)
            ((5, 8, 3), [0, 3, 6, 8], (5, 3, 3)),
            ((2, 10, 4), [0, 2, 5, 7, 10], (2, 4, 4)),
        ]
        
        for src_shape, indptr_values, expected_shape in test_cases:
            src = np.random.randn(*src_shape).astype(np.float32)
            indptr = np.array(indptr_values, dtype=np.int64).reshape(1, -1)
            inputs = (src, indptr)
            
            # Calculate expected PyTorch output for validation
            pytorch_out = model(torch.tensor(src), torch.tensor(indptr))
            assert pytorch_out.shape == expected_shape, f"PyTorch output shape {pytorch_out.shape} doesn't match expected {expected_shape}"
            
            self._test(model, None, "torch_scatter::segment_mean_csr", ie_device, 
                      precision, ir_version, 
                      custom_test_inputs=[inputs])
