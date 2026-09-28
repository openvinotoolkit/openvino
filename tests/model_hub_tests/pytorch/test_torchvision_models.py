# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform

import pytest
import torch
from torchvision.models import list_models, get_model
from models_hub_common.utils import get_models_list, retry

from torch_utils import TestTorchConvertModel


def get_all_models() -> list:
    m_list = list_models()
    return m_list


# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestTorchHubConvertModel(TestTorchConvertModel):
    @retry(3, exceptions=(OSError,), delay=1)
    def load_model(self, model_name, model_link):
        m = get_model(model_name, weights='DEFAULT')
        m.eval()
        if model_name == "s3d" or any(m in model_name for m in ["swin3d", "r3d_18", "mc3_18", "r2plus1d_18"]):
            self.example = (torch.randn([2, 3, 224, 224, 224]),)
            self.inputs = (torch.randn([3, 3, 224, 224, 224]),)
        elif "mvit" in model_name:
            # 16 frames from video
            self.example = (torch.randn(2, 3, 16, 224, 224),)
            self.inputs = (torch.randn(3, 3, 16, 224, 224),)
        elif "raft" in model_name:
            # torchvision.io.read_video was removed in torchvision 0.27.0 and the
            # CPU wheel has no video-reading backend.  Use smooth synthetic frames
            # (sine/cosine patterns with a small periodic shift) instead: structured
            # inputs give RAFT a well-conditioned optical-flow problem so the FP32
            # differences between OV and PyTorch stay within the 0.05 tolerance.
            # The shape (2, 3, 520, 960) matches what the old resize+normalise
            # pipeline produced from the basketball clip.
            import math
            h, w = 520, 960
            y = torch.linspace(-math.pi, math.pi, h).view(h, 1)
            x = torch.linspace(-math.pi, math.pi, w).view(1, w)
            freq = 4.0
            frame = torch.stack([
                torch.sin(freq * x) * torch.cos(freq * y),
                torch.cos(freq * x) * torch.sin(freq * y),
                torch.sin(freq * (x + y) * 0.5),
            ], dim=0)  # [3, 520, 960], values in [-1, 1]
            # 2-pixel cyclic roll along width = stable constant horizontal flow
            frame1 = frame.unsqueeze(0).expand(2, -1, -1, -1).contiguous()
            frame2 = torch.roll(frame, shifts=2, dims=2).unsqueeze(0).expand(2, -1, -1, -1).contiguous()
            self.example = (frame1, frame2)
            self.inputs = (frame1.clone(), frame2.clone())
        elif "vit_h_14" in model_name:
            self.example = (torch.randn(1, 3, 518, 518),)
            self.inputs = (torch.randn(1, 3, 518, 518),)
        elif model_name.startswith("vit_"):
            self.example = (torch.randn(1, 3, 224, 224),)
            self.inputs = (torch.randn(1, 3, 224, 224),)
        else:
            self.example = (torch.randn(2, 3, 224, 224),)
            self.inputs = (torch.randn(3, 3, 224, 224),)
        if (getattr(self, "mode", None) == "export"
                and "raft" not in model_name
                and not model_name.startswith("vit_")):
            from openvino import PartialShape, Dimension
            shape = list(self.example[0].shape)
            shape[0] = Dimension(1, 3)
            self.dynamo_input = (PartialShape(shape),)
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

    def get_supported_precommit_models():
        models = [
            "efficientnet_b7",
        ]
        if platform.machine() not in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64']:
            models.extend([
                "raft_small",
                "swin_v2_s",
                "quantized_mobilenet_v3_large",
            ])
        return models

    @pytest.mark.parametrize("model_name", get_supported_precommit_models())
    @pytest.mark.precommit
    def test_convert_model_precommit(self, model_name, ie_device):
        self.mode = "trace"
        self.run(model_name, None, ie_device)

    @pytest.mark.parametrize("model_name", ["efficientnet_b7"])
    @pytest.mark.precommit
    def test_convert_model_precommit_export(self, model_name, ie_device):
        self.mode = "export"
        self.run(model_name, None, ie_device)

    @pytest.mark.parametrize("name,link,mark,reason", get_models_list(os.path.join(os.path.dirname(__file__), "torchvision_models")))
    @pytest.mark.parametrize("mode", ["trace", "export"])
    @pytest.mark.nightly
    def test_convert_model_all_models(self, mode, name, link, mark, reason,  ie_device):
        self.mode = mode
        assert mark is None or mark in [
            'skip', 'xfail', 'xfail_trace', 'xfail_export'], f"Incorrect test case for {name}"
        if mark == 'skip':
            pytest.skip(reason)
        elif mark in ['xfail', f'xfail_{mode}']:
            pytest.xfail(reason)
        self.run(name, None, ie_device)
