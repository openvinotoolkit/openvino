# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import tempfile
from contextlib import contextmanager
from types import MethodType

import pytest
import torch
import torch.nn.functional as F

from torch_utils import TestTorchConvertModel


MODEL_INPUTS = [
    ("aliked-t16", (1, 3, 256, 320)),
    ("aliked-n16", (1, 3, 256, 320)),
    ("aliked-n16rot", (1, 3, 256, 320)),
    ("aliked-n32", (1, 3, 256, 320)),
    ("DaD", (1, 3, 256, 320)),
    ("DaD", (1, 3, 255, 319)),
    ("RaCo", (1, 3, 256, 320)),
    ("RaCo", (1, 3, 255, 319)),
    ("RaCo", (1, 1, 256, 320)),
    ("DISK", (1, 3, 256, 320)),
    ("DISK", (1, 3, 255, 319)),
    ("XFeat", (1, 3, 256, 320)),
    ("XFeat", (1, 1, 255, 319)),
    ("DeDoDe", (1, 3, 256, 320)),
    ("DeDoDe", (1, 3, 255, 319)),
]


def xfeat_nms(self, image, threshold=0.05, kernel_size=5):
    # Batch-one inference avoids padding to a Python length derived from detected points.
    assert image.shape[0] == 1
    maxima = F.max_pool2d(image, kernel_size, stride=1, padding=kernel_size // 2)
    positions = (image == maxima) & (image > threshold)
    return positions[0].nonzero()[..., 1:].flip(-1).unsqueeze(0)


@contextmanager
def isolated_package(package):
    # Model repositories use generic package names that conflict with other model-hub repositories.
    def is_package_module(name):
        return name == package or name.startswith(f"{package}.")

    saved_modules = {name: module for name, module in list(sys.modules.items()) if is_package_module(name)}
    for name in saved_modules:
        sys.modules.pop(name)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if is_package_module(name):
                sys.modules.pop(name)
        sys.modules.update(saved_modules)


def load_xfeat():
    with isolated_package("modules"):
        return torch.hub.load(
            "verlab/accelerated_features:e92685f57f8318b18725c5c8c0bd28c7fe188d9a",
            "XFeat",
            trust_repo=True,
            skip_validation=True,
            top_k=128,
        )


def load_aliked(repo_dir, model_name):
    sys.path.insert(0, repo_dir)
    try:
        with isolated_package("nets"), isolated_package("custom_ops"):
            # Use the PyTorch get_patches fallback instead of the uncompiled extension.
            sys.modules["custom_ops"] = None
            from nets.aliked import ALIKED

            return ALIKED(model_name=model_name, device="cpu", top_k=128)
    finally:
        sys.path.remove(repo_dir)


class KeypointDetector(torch.nn.Module):
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name

    def forward(self, image):
        if self.model_name.startswith("aliked-"):
            # Original forward synchronizes CUDA for timing, so run its stages directly.
            feature_map, score_map = self.model.extract_dense_map(image)
            keypoints, scores, _ = self.model.dkd(score_map)
            descriptors, _ = self.model.desc_head(feature_map, keypoints)
            return {
                "keypoints": torch.stack(keypoints),
                "keypoint_scores": torch.stack(scores),
                "descriptors": torch.stack(descriptors),
            }
        if self.model_name == "DISK":
            features = self.model(image, n=128, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)
            return {
                "keypoints": torch.stack([f.keypoints for f in features]),
                "keypoint_scores": torch.stack([f.detection_scores for f in features]),
                "descriptors": torch.stack([f.descriptors for f in features]),
            }
        if self.model_name == "DaD":
            return self.model.detect(
                {"image": self.model.normalizer(image)},
                num_keypoints=128,
                return_dense_probs=True,
            )
        if self.model_name == "XFeat":
            return self.model.detectAndCompute(image)[0]
        if self.model_name == "DeDoDe":
            keypoints, scores, descriptors = self.model(image, n=128)
            return {"keypoints": keypoints, "keypoint_scores": scores, "descriptors": descriptors}
        return self.model({"image": image})


class TestKeypointDetectorsConvertModel(TestTorchConvertModel):
    ov_config = {"INFERENCE_PRECISION_HINT": "f32"}
    aliked_repo = None

    def teardown_class(self):
        if self.aliked_repo is not None:
            self.aliked_repo.cleanup()
            self.aliked_repo = None

    def get_aliked_repo(self):
        cls = type(self)
        if cls.aliked_repo is None:
            repo = tempfile.TemporaryDirectory()
            subprocess.check_call(["git", "clone", "-q", "https://github.com/Shiaoming/ALIKED.git", repo.name])
            subprocess.check_call(
                ["git", "checkout", "-q", "683d7c65197395c0b3f01ebe76e1084a27e73a65"], cwd=repo.name)
            cls.aliked_repo = repo
        return cls.aliked_repo.name

    def load_model(self, model_name, model_args):
        aliked_repo_dir = None
        if model_name.startswith("aliked-"):
            input_shape, aliked_repo_dir = model_args
        else:
            input_shape = model_args

        if model_name.startswith("aliked-"):
            model = load_aliked(aliked_repo_dir, model_name)
        elif model_name == "DaD":
            import dad

            model = dad.load_DaD()
            # Compare FP32 inference without the detector's automatic mixed precision.
            for module in model.modules():
                if hasattr(module, "amp"):
                    module.amp = False
        elif model_name == "RaCo":
            model = torch.hub.load(
                "cvg/RaCo:35790eb48074ed14839d0fb496b8806caa4e766b",
                "raco",
                trust_repo=True,
                skip_validation=True,
                max_num_keypoints=128,
            )
        elif model_name == "DISK":
            from kornia.feature import DISK

            model = DISK.from_pretrained("depth", device=torch.device("cpu"))
        elif model_name == "XFeat":
            model = load_xfeat()
            model.dev = torch.device("cpu")
            model.NMS = MethodType(xfeat_nms, model)
        elif model_name == "DeDoDe":
            from kornia.feature import DeDoDe

            model = DeDoDe.from_pretrained(
                detector_weights="L-C4-v2", descriptor_weights="B-upright", amp_dtype=torch.float32)
            for module in model.modules():
                if hasattr(module, "amp"):
                    module.amp = False
        else:
            raise ValueError(f"Unknown keypoint detector: {model_name}")

        if model_name in ("DaD", "RaCo"):
            from torchvision.transforms.v2 import Normalize

            normalizer = model.normalizer
            # v2 checks the fixed standard deviations without tensor-dependent branching.
            model.normalizer = Normalize(normalizer.mean, normalizer.std)

        generator = torch.Generator().manual_seed(0)
        self.example = (torch.rand(input_shape, generator=generator),)
        self.inputs = (torch.rand(input_shape, generator=generator),)
        return KeypointDetector(model, model_name).cpu().eval()

    @pytest.mark.nightly
    @pytest.mark.parametrize("name,input_shape,mode", [
        pytest.param(
            name, input_shape, mode,
            id=f"{name}-{input_shape[1]}ch-{input_shape[2]}x{input_shape[3]}-{mode}",
            marks=pytest.mark.precommit if name == "aliked-n16rot" and mode == "trace" else (),
        )
        for name, input_shape in MODEL_INPUTS
        for mode in ("trace", "export")
    ])
    def test_convert_model(self, name, input_shape, mode, ie_device):
        self.mode = mode
        model_args = (input_shape, self.get_aliked_repo()) if name.startswith("aliked-") else input_shape
        self.run(name, model_args, ie_device)
