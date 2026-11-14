# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import MethodType

import pytest
import torch
import torch.nn.functional as F
from openvino import convert_model

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
    ("SuperPoint", (1, 3, 256, 320)),
    ("SuperPoint", (1, 1, 255, 319)),
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


def load_xfeat():
    # XFeat's generic package name conflicts with other model-hub repositories.
    saved_modules = {name: module for name, module in list(sys.modules.items())
                     if name == "modules" or name.startswith("modules.")}
    for name in saved_modules:
        sys.modules.pop(name)
    try:
        return torch.hub.load(
            "verlab/accelerated_features:e92685f57f8318b18725c5c8c0bd28c7fe188d9a",
            "XFeat",
            trust_repo=True,
            skip_validation=True,
            top_k=128,
        )
    finally:
        for name in list(sys.modules):
            if name == "modules" or name.startswith("modules."):
                sys.modules.pop(name)
        sys.modules.update(saved_modules)


class KeypointDetector(torch.nn.Module):
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name

    def forward(self, image):
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
        outputs = self.model({"image": image})
        if self.model_name == "SuperPoint":
            return outputs["keypoints"], outputs["keypoint_scores"], outputs["descriptors"]
        return outputs


class TestKeypointDetectorsConvertModel(TestTorchConvertModel):
    ov_config = {"INFERENCE_PRECISION_HINT": "f32"}

    def load_model(self, model_name, input_shape):
        if model_name.startswith("aliked-"):
            from lightglue import ALIKED

            model = ALIKED(model_name=model_name,
                           max_num_keypoints=128,
                           detection_threshold=-1)
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
        elif model_name == "SuperPoint":
            from lightglue import SuperPoint

            # The uncapped configuration avoids Python branching on the detection count.
            model = SuperPoint(max_num_keypoints=None)
        elif model_name == "DISK":
            from lightglue import DISK

            model = DISK(max_num_keypoints=128)
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

    def convert_model_impl(self, model_obj):
        if getattr(self, "mode", None) == "export" and model_obj.model_name in ("DaD", "DeDoDe", "DISK"):
            exported = torch.export.export(model_obj, self.example)
            # Decompose autocast regions and preserve DISK's padding through functionalization.
            return convert_model(exported.run_decompositions())
        return super().convert_model_impl(model_obj)

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
        self.run(name, input_shape, ie_device)
