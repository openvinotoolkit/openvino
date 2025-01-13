# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import tempfile

import pytest
import torch
import torchvision
from packaging import version

from torch_utils import TestTorchConvertModel
from openvino import convert_model
import numpy as np

# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestGFPGANConvertModel(TestTorchConvertModel):
    def setup_class(self):
        self.repo_dir = tempfile.TemporaryDirectory()
        os.system(
            f"git clone https://github.com/TencentARC/GFPGAN.git {self.repo_dir.name}")
        subprocess.check_call(
            ["git", "checkout", "bc5a5deb95a4a9653851177985d617af1b9bfa8b"], cwd=self.repo_dir.name)
        checkpoint_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
        subprocess.check_call(
            ["wget", "-nv", checkpoint_url], cwd=self.repo_dir.name)

    def load_model(self, model_name, model_link):
        sys.path.append(self.repo_dir.name)
        from gfpgan import GFPGANer

        filename = os.path.join(self.repo_dir.name, 'GFPGANv1.3.pth')
        arch = 'clean'
        channel_multiplier = 2
        restorer = GFPGANer(
            model_path=filename,
            upscale=2,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=None)

        self.example = (torch.randn(1, 3, 512, 512),)
        self.inputs = (torch.randn(1, 3, 512, 512),)
        return restorer.gfpgan

    def convert_model(self, model_obj):
        ov_model = convert_model(
            model_obj, example_input=self.example, input=[1, 3, 512, 512], verbose=True)
        return ov_model

    def compare_results(self, fw_outputs, ov_outputs):
        assert len(fw_outputs) == len(ov_outputs), \
            "Different number of outputs between framework and OpenVINO:" \
            " {} vs. {}".format(len(fw_outputs), len(ov_outputs))

        fw_eps = 5e-2
        is_ok = True
        for i in range(len(ov_outputs)):
            cur_fw_res = fw_outputs[i]
            cur_ov_res = ov_outputs[i]
            try:
                np.testing.assert_allclose(
                    cur_ov_res, cur_fw_res, fw_eps, fw_eps)
            except AssertionError as e:
                print(e)
                # The model has aten::normal_ operation which produce random numbers.
                #  Cannot reliably validate the output 0
                if i != 0:
                    is_ok = False
        assert is_ok, "Accuracy validation failed"

    def teardown_class(self):
        # remove all downloaded files from cache
        self.repo_dir.cleanup()

    @pytest.mark.skipif(version.parse(torchvision.__version__) >= version.parse("0.17"),
                        reason="torchvision==0.17 have removed module torchvision.transforms.functional_tensor which is required by GFPGAN")
    def test_convert_model(self, ie_device):
        self.run("GFPGAN", None, ie_device)
