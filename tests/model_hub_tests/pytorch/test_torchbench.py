# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
import pytest
import torch
import tempfile
from torch_utils import process_pytest_marks, get_models_list, TestTorchConvertModel


# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestTorchbenchmarkConvertModel(TestTorchConvertModel):
    _model_list_path = os.path.join(
        os.path.dirname(__file__), "torchbench_models")

    def setup_class(self):
        super().setup_class(self)
        # sd model doesn't need token but torchbench need it to be specified
        os.environ['HUGGING_FACE_HUB_TOKEN'] = 'x'
        torch.set_grad_enabled(False)

        self.infer_timeout = 800

        self.repo_dir = tempfile.TemporaryDirectory()
        os.system(
            f"git clone https://github.com/pytorch/benchmark.git {self.repo_dir.name}")
        subprocess.check_call(
            ["git", "checkout", "dbc109791dbb0dfb58775a5dc284fc2c3996cb30"], cwd=self.repo_dir.name)

    def load_model(self, model_name, model_link):
        subprocess.check_call([sys.executable, "install.py"] + [model_name], cwd=self.repo_dir.name)
        sys.path.append(self.repo_dir.name)
        from torchbenchmark import load_model_by_name
        try:
            model_cls = load_model_by_name(
                model_name)("eval", "cpu", jit=False)
        except:
            model_cls = load_model_by_name(model_name)("eval", "cpu")
        model, self.example = model_cls.get_module()
        self.inputs = self.example
        # initialize selected models
        if model_name in ["BERT_pytorch", "yolov3"]:
            model(*self.example)
        return model

    def teardown_class(self):
        # cleanup tmpdir
        self.repo_dir.cleanup()

    @pytest.mark.parametrize("name", process_pytest_marks(_model_list_path))
    @pytest.mark.nightly
    def test_convert_model_all_models(self, name, ie_device):
        self.run(name, None, ie_device)
