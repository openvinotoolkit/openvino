# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform
import pytest
import torch

from torch_utils import TestTorchConvertModel

# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestEasyOCRConvertModel(TestTorchConvertModel):
    def load_model(self, model_name, model_link):
        import easyocr
        if model_name == "detector":
            model = easyocr.Reader(["en"], quantize=False).detector
            self.example = (torch.rand(1, 3, 608, 800),)
            self.inputs = (torch.rand(1, 3, 608, 800),)
        elif model_name == "recognizer":
            model = easyocr.Reader(["en"], quantize=False).recognizer
            self.example = (torch.rand(1, 1, 64, 320), torch.rand(1, 33))
            self.inputs = (torch.rand(1, 1, 64, 320), torch.rand(1, 33))
        else:
            raise RuntimeError("Unknown model type")
        return model

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("name", ["detector", "recognizer"])
    def test_convert_model(self, name, ie_device):
        if platform.machine() in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64']:
            pytest.skip("EasyOCR models are not enabled on ARM")
        self.run(name, None, ie_device)
