# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import tempfile

import pytest
import torch
import yaml

from torch_utils import TestTorchConvertModel

# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestThinPlateSplineMotionModel(TestTorchConvertModel):
    def setup_class(self):
        self.repo_dir = tempfile.TemporaryDirectory()
        os.system(
            f"git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.git {self.repo_dir.name}")
        subprocess.check_call(
            ["git", "checkout", "c616878812c9870ed81ac72561be2676fd7180e2"], cwd=self.repo_dir.name)
        # verify model on random weights

    def load_model(self, model_name, model_link):
        sys.path.append(self.repo_dir.name)
        from modules.inpainting_network import InpaintingNetwork
        from modules.keypoint_detector import KPDetector
        from modules.dense_motion import DenseMotionNetwork

        class Mixer(torch.nn.Module):
            def __init__(self, config_path):
                super(Mixer, self).__init__()

                with open(config_path) as f:
                    config = yaml.full_load(f)

                self.inpainting_network = InpaintingNetwork(**config['model_params']['generator_params'],
                                                            **config['model_params']['common_params'])
                self.kp_detector = KPDetector(
                    **config['model_params']['common_params'])
                self.dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                                               **config['model_params']['dense_motion_params'])

            def forward(self, source, driving):
                kp_source = self.kp_detector(source)
                kp_driving = self.kp_detector(driving)

                dense_motion = self.dense_motion_network(
                    source_image=source, kp_driving=kp_driving, kp_source=kp_source, bg_param=None, dropout_flag=False)
                out = self.inpainting_network(source, dense_motion)
                return out["prediction"]

        config_path = os.path.join(
            self.repo_dir.name, "config", "vox-256.yaml")
        m = Mixer(config_path)
        self.example = (torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256))
        self.inputs = (torch.rand(1, 3, 256, 256), torch.rand(1, 3, 256, 256))
        return m

    def teardown_class(self):
        # remove all downloaded files from cache
        self.repo_dir.cleanup()

    @pytest.mark.nightly
    def test_convert_model(self, ie_device):
        self.run("tpsmm", None, ie_device)
