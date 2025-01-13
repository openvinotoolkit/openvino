# Copyright (C) 2018-2025 Intel Corporation
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


class TestSpeechTransformerConvertModel(TestTorchConvertModel):
    def setup_class(self):
        self.repo_dir = tempfile.TemporaryDirectory()
        os.system(
            f"git clone https://github.com/mvafin/Speech-Transformer.git {self.repo_dir.name}")
        subprocess.check_call(["git", "checkout", "071eebb7549b66bae2cb93e3391fe99749389456"], cwd=self.repo_dir.name)
        checkpoint_url = "https://github.com/foamliu/Speech-Transformer/releases/download/v1.0/speech-transformer-cn.pt"
        subprocess.check_call(["wget", "-nv", checkpoint_url], cwd=self.repo_dir.name)

    def load_model(self, model_name, model_link):
        sys.path.append(self.repo_dir.name)
        from transformer.transformer import Transformer

        filename = os.path.join(self.repo_dir.name, 'speech-transformer-cn.pt')
        m = Transformer()
        m.load_state_dict(torch.load(
            filename, map_location=torch.device('cpu')))

        self.example = (torch.randn(32, 209, 320),
                        torch.stack(sorted(torch.randint(55, 250, [32]), reverse=True)),
                        torch.randint(-1, 4232, [32, 20]))
        self.inputs = (torch.randn(32, 209, 320),
                       torch.stack(sorted(torch.randint(55, 400, [32]), reverse=True)),
                       torch.randint(-1, 4232, [32, 25]))
        return m

    def infer_fw_model(self, model_obj, inputs):
        fw_outputs = model_obj(*[torch.from_numpy(i) for i in inputs])
        if isinstance(fw_outputs, dict):
            for k in fw_outputs.keys():
                fw_outputs[k] = fw_outputs[k].numpy(force=True)
        elif isinstance(fw_outputs, (list, tuple)):
            fw_outputs = [o.numpy(force=True) for o in fw_outputs]
        else:
            fw_outputs = [fw_outputs.numpy(force=True)]
        return fw_outputs

    def teardown_class(self):
        # remove all downloaded files from cache
        self.repo_dir.cleanup()

    @pytest.mark.nightly
    def test_convert_model(self, ie_device):
        self.run("speech-transformer", None, ie_device)
