# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest
from torchvision.ops import roi_align


class TestROIAlign(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.boxes)

    def create_model(self, output_size, spatial_scale, sampling_ratio, aligned):

        class torchvision_roi_align(torch.nn.Module):
            def __init__(self, output_size, spatial_scale, sampling_ratio, aligned):
                super().__init__()
                self.output_size = output_size
                self.spatial_scale = spatial_scale
                self.sampling_ratio = sampling_ratio
                self.aligned = aligned

            def forward(self, input_tensor, rois):
                return roi_align(
                    input_tensor,
                    rois.to(dtype=input_tensor.dtype),
                    self.output_size,
                    self.spatial_scale,
                    self.sampling_ratio,
                    self.aligned,
                )

        ref_net = None

        return (torchvision_roi_align(output_size, spatial_scale, sampling_ratio, aligned),
                ref_net, "torchvision::roi_align")

    @pytest.mark.parametrize('input_shape', [
        [4, 5, 6, 7],
    ])
    @pytest.mark.parametrize('boxes', (np.array([[1, 2, 2, 3, 3]]).astype(np.float32),
                                       np.array([[0, 1, 2, 5, 4],
                                                 [2, 1, 2, 5, 4],
                                                 [3, 1, 2, 5, 4]]).astype(np.float32)))
    @pytest.mark.parametrize('output_size', ((4, 5), (3, 2), 3))
    @pytest.mark.parametrize('spatial_scale', (0.5, 1.0))
    @pytest.mark.parametrize('sampling_ratio', (0, 1))
    @pytest.mark.parametrize('aligned', (True, False))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_roi_align(self, ie_device, precision, ir_version, input_shape, boxes, output_size,
                       spatial_scale, sampling_ratio, aligned):
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self.boxes = boxes
        self._test(*self.create_model(output_size, spatial_scale, sampling_ratio, aligned),
                   ie_device, precision, ir_version, trace_model=True)
