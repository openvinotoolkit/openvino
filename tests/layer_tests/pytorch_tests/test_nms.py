# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
from pytorch_layer_test_class import PytorchLayerTest
import numpy as np
import torch
import torchvision


@pytest.mark.parametrize('boxes_num', (1, 2, 3, 4, 5))
class TestNms(PytorchLayerTest):

    def _prepare_input(self):
        # PyTorch requires that boxes are in (x1, y1, x2, y2) format, where 0<=x1<x2 and 0<=y1<y2
        boxes = np.array([[np.random.uniform(1, 3), np.random.uniform(2, 6),
                           np.random.uniform(4, 6), np.random.uniform(7, 9)] for _ in range(self.boxes_num)]).astype(np.float32)
        # scores can be negative
        scores = np.random.randn(self.boxes_num).astype(np.float32)
        return (boxes, scores)

    def create_model(self):
        class torchvision_nms(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.iou_threshold = 0.5

            def forward(self, boxes, scores):
                return torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=self.iou_threshold)

        ref_net = None

        return torchvision_nms(), ref_net, "torchvision::nms"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_nms(self, ie_device, precision, ir_version, boxes_num):
        self.boxes_num = boxes_num
        self._test(*self.create_model(), ie_device, precision, ir_version)
