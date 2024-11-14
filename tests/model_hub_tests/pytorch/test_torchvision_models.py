# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import pytest
import torch
import torchvision.transforms.functional as F
from torchvision.models import list_models, get_model, get_model_weights
from models_hub_common.utils import get_models_list, retry

from torch_utils import TestTorchConvertModel


def get_all_models() -> list:
    m_list = list_models()
    return m_list


def get_video():
    """
    Download video and return frames.
    Using free video from pexels.com, credits go to Pavel Danilyuk.
    Initially used in https://pytorch.org/vision/stable/auto_examples/plot_optical_flow.html
    """
    from pathlib import Path
    from urllib.request import urlretrieve
    from torchvision.io import read_video

    video_url = "https://download.pytorch.org/tutorial/pexelscom_pavel_danilyuk_basketball_hd.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "basketball.mp4"
        _ = urlretrieve(video_url, video_path)

        frames, _, _ = read_video(str(video_path), output_format="TCHW")
    return frames


def prepare_frames_for_raft(name, frames1, frames2):
    w = get_model_weights(name).DEFAULT
    img1_batch = torch.stack(frames1)
    img2_batch = torch.stack(frames2)
    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    img1_batch, img2_batch = w.transforms()(img1_batch, img2_batch)
    return (img1_batch, img2_batch)


# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestTorchHubConvertModel(TestTorchConvertModel):
    @retry(3, exceptions=(OSError,), delay=1)
    def load_model(self, model_name, model_link):
        m = get_model(model_name, weights='DEFAULT')
        m.eval()
        if model_name == "s3d" or any([m in model_name for m in ["swin3d", "r3d_18", "mc3_18", "r2plus1d_18"]]):
            self.example = (torch.randn([1, 3, 224, 224, 224]),)
            self.inputs = (torch.randn([1, 3, 224, 224, 224]),)
        elif "mvit" in model_name:
            # 16 frames from video
            self.example = (torch.randn(1, 3, 16, 224, 224),)
            self.inputs = (torch.randn(1, 3, 16, 224, 224),)
        elif "raft" in model_name:
            frames = get_video()
            self.example = prepare_frames_for_raft(model_name,
                                                   [frames[100], frames[150]],
                                                   [frames[101], frames[151]])
            self.inputs = prepare_frames_for_raft(model_name,
                                                  [frames[75], frames[125]],
                                                  [frames[76], frames[126]])
        elif "vit_h_14" in model_name:
            self.example = (torch.randn(1, 3, 518, 518),)
            self.inputs = (torch.randn(1, 3, 518, 518),)
        else:
            self.example = (torch.randn(1, 3, 224, 224),)
            self.inputs = (torch.randn(1, 3, 224, 224),)
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

    @pytest.mark.parametrize("model_name", ["efficientnet_b7",
                                            "raft_small",
                                            "swin_v2_s",
                                            "quantized_mobilenet_v3_large",
                                            ])
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
