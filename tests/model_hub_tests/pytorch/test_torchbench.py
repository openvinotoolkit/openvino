# Copyright (C) 2018-2026 Intel Corporation
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
            ["git", "checkout", "364420aeca07d9519840a5b6e771035e4dff9d72"], cwd=self.repo_dir.name)
        # torchaudio has no builds for recent torch and tested models don't need it
        deps_file = os.path.join(self.repo_dir.name, "utils", "__init__.py")
        with open(deps_file) as f:
            deps = f.read()
        with open(deps_file, "w") as f:
            f.write(deps.replace('"torchvision", "torchaudio"]', '"torchvision"]'))
        # some model deps (e.g. visdom) are sdists importing pkg_resources, removed in setuptools 81
        build_constraints = os.path.join(self.repo_dir.name, "build_constraints.txt")
        with open(build_constraints, "w") as f:
            f.write("setuptools<81\n")
        # pip>=25.3 uses PIP_BUILD_CONSTRAINT for isolated builds, older pip uses PIP_CONSTRAINT
        os.environ["PIP_BUILD_CONSTRAINT"] = build_constraints
        os.environ["PIP_CONSTRAINT"] = build_constraints
        # build legacy setup.py packages in isolation, so setuptools plugins from the test env
        # (e.g. kernels egg_info writer) are not loaded
        os.environ["PIP_USE_PEP517"] = "1"

    def load_model(self, model_name, model_link):
        subprocess.check_call([sys.executable, "install.py"] + [model_name], cwd=self.repo_dir.name)
        sys.path.append(self.repo_dir.name)
        import numpy as np
        if int(np.__version__.split(".")[0]) >= 2:
            # maml inputs are pickled with numpy<2 path, which weights_only load doesn't map to numpy._core
            torch.serialization.add_safe_globals(
                [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct")])
        from torchbenchmark import load_model_by_name
        try:
            model_cls = load_model_by_name(
                model_name)("eval", "cpu", jit=False)
        except TypeError:
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
