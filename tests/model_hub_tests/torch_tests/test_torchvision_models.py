# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
import tempfile
import torchvision.transforms.functional as F
from openvino import convert_model
from models_hub_common.test_convert_model import TestConvertModel
from models_hub_common.utils import get_models_list
from torch.export import export
from torch.fx.experimental.proxy_tensor import make_fx
from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder


def get_all_models() -> list:
    m_list = torch.hub.list("pytorch/vision", skip_validation=True)
    m_list.remove("get_model_weights")
    m_list.remove("get_weight")
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
    w = torch.hub.load("pytorch/vision", "get_model_weights",
                       name=name, skip_validation=True).DEFAULT
    img1_batch = torch.stack(frames1)
    img2_batch = torch.stack(frames2)
    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    img1_batch, img2_batch = w.transforms()(img1_batch, img2_batch)
    return (img1_batch, img2_batch)


# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestTorchHubConvertModel(TestConvertModel):
    def setup_class(self):
        self.cache_dir = tempfile.TemporaryDirectory()
        # set temp dir for torch cache
        if os.environ.get('USE_SYSTEM_CACHE', 'True') == 'False':
            torch.hub.set_dir(str(self.cache_dir.name))

    def load_model(self, model_name, model_link):
        m = torch.hub.load("pytorch/vision", model_name,
                           weights='DEFAULT', skip_validation=True)
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
        else:
            self.example = (torch.randn(1, 3, 224, 224),)
            self.inputs = (torch.randn(1, 3, 224, 224),)
        return m

    def get_inputs_info(self, model_obj):
        return None

    def prepare_inputs(self, inputs_info):
        return [i.numpy() for i in self.inputs]

    def convert_model(self, model_obj):
        #ov_model = convert_model(model_obj, example_input=self.example)
        em = export(model_obj, self.example)
        mfx = make_fx(em)(*self.example)
        mfx.eval()
        gm = mfx
        
        input_shapes = []
        input_types = []
        for input_data in self.example:
            input_types.append(input_data.type())
            input_shapes.append(input_data.size())
            
        decoder = TorchFXPythonDecoder(gm, gm, input_shapes=input_shapes, input_types=input_types)
        
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework("pytorch")
        im = fe.load(decoder)
        with torch.no_grad():
            ov_model = fe.convert(im)
        params = ov_model.get_parameters()
        if params[-1].get_element_type().is_dynamic():
            ov_model.remove_parameter(params[-1])
        print(ov_model)
        return ov_model

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

    def teardown_method(self):
        # cleanup tmpdir
        self.cache_dir.cleanup()
        super().teardown_method()

    @pytest.mark.parametrize("model_name", ["efficientnet_b7", "raft_small", "swin_v2_s"])
    @pytest.mark.precommit
    def test_convert_model_precommit(self, model_name, ie_device):
        self.run(model_name, None, ie_device)

    @pytest.mark.parametrize("name",
                             [pytest.param(n, marks=pytest.mark.xfail(reason=r)) if m == "xfail" else n for n, _, m, r in get_models_list(os.path.join(os.path.dirname(__file__), "torchvision_models"))])
    @pytest.mark.nightly
    def test_convert_model_all_models(self, name, ie_device):
        self.run(name, None, ie_device)
