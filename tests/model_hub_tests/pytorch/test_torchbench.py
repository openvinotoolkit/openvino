# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import tempfile

import pytest
import torch

from torch_utils import get_models_list, TestTorchConvertModel


# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestTorchbenchmarkConvertModel(TestTorchConvertModel):
    infer_timeout = 800
    _model_list_path = os.path.join(
        os.path.dirname(__file__), "torchbench_models")

    def setup_class(self):
        super().setup_class(self)
        self.installed_models = set()
        self.repo_dir = tempfile.TemporaryDirectory()
        subprocess.check_call([
            "git", "clone", "https://github.com/pytorch/benchmark.git", self.repo_dir.name])
        subprocess.check_call(
            ["git", "checkout", "88e8837b7323627eb874dedff0c3e4e494d36ff8"], cwd=self.repo_dir.name)
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
        sys.path.insert(0, self.repo_dir.name)

    def load_model(self, model_name, model_link):
        import numpy as np
        if int(np.__version__.split(".")[0]) >= 2:
            # maml inputs are pickled with numpy<2 path, which weights_only load doesn't map to numpy._core
            torch.serialization.add_safe_globals(
                [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct")])
        from torchbenchmark import load_model_by_name
        benchmark = load_model_by_name(model_name)(test="eval", device="cpu")
        model, self.example = benchmark.get_module()
        if model_name in ("hf_Whisper", "hf_distil_whisper"):
            model.config.return_dict = False
        if model_name in ("BERT_pytorch", "yolov3"):
            model(*self.example)
        return model

    def infer_fw_model(self, model_obj, inputs):
        outputs = super().infer_fw_model(model_obj, inputs)
        if isinstance(outputs, dict):
            return list(outputs.values())
        return outputs

    def teardown_class(self):
        sys.path.remove(self.repo_dir.name)
        self.repo_dir.cleanup()

    @pytest.mark.parametrize("name,link,mark,reason", get_models_list(_model_list_path))
    @pytest.mark.parametrize("mode", ["trace", "export"])
    @pytest.mark.nightly
    def test_convert_model_all_models(self, name, link, mark, reason, mode, ie_device, request):
        marks = mark.split(";") if mark else []
        assert all(m in ("skip", "skip_trace", "skip_export", "xfail", "xfail_trace", "xfail_export")
                   for m in marks), f"Incorrect test case for {name}"
        if "skip" in marks or f"skip_{mode}" in marks:
            pytest.skip(reason)
        if name not in self.installed_models:
            subprocess.check_call([sys.executable, "install.py", name], cwd=self.repo_dir.name,
                                  timeout=self.infer_timeout)
            self.installed_models.add(name)
        if "xfail" in marks or f"xfail_{mode}" in marks:
            request.node.add_marker(pytest.mark.xfail(reason=reason))
        self.mode = mode
        self.run(name, link, ie_device)
