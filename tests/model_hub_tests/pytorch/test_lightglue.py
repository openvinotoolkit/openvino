# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import tempfile

import pytest
import torch

from torch_utils import TestTorchConvertModel

# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestLightGlueModel(TestTorchConvertModel):
    def setup_class(self):
        self.repo_dir = tempfile.TemporaryDirectory()
        os.system(
            f"git clone https://github.com/fabio-sim/LightGlue-ONNX {self.repo_dir.name}")
        subprocess.check_call(
            ["git", "checkout", "1a9d150ae59fa83ac6c97a7cf181cdfec5e1c4f9"], cwd=self.repo_dir.name)

    def load_model(self, model_name, extractor_type):
        sys.path.append(self.repo_dir.name)
        from lightglue_onnx import LightGlue

        m = LightGlue(extractor_type)

        def get_example():
            dummy_kpts0 = torch.randn(size=[1, 1580, 2], dtype=torch.float32)
            dummy_kpts1 = torch.randn(size=[1, 3401, 2], dtype=torch.float32)
            dummy_desc0 = torch.randn(size=[1, 1580, 256], dtype=torch.float32)
            dummy_desc1 = torch.randn(size=[1, 3401, 256], dtype=torch.float32)
            return (dummy_kpts0, dummy_kpts1, dummy_desc0, dummy_desc1)
        self.example = get_example()
        self.inputs = get_example()
        return m

    def teardown_class(self):
        # remove all downloaded files from cache
        self.repo_dir.cleanup()

    @pytest.mark.nightly
    def test_convert_model(self, ie_device):
        self.run("lightglue", "superpoint", ie_device)
