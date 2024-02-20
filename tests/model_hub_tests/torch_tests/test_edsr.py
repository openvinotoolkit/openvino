# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import random
import torch
from models_hub_common.constants import hf_hub_cache_dir
from models_hub_common.utils import cleanup_dir

from torch_utils import TestTorchConvertModel
from super_image import ImageLoader, EdsrModel, MsrnModel, A2nModel, PanModel, CarnModel, DrlnModel, MdsrModel, HanModel, AwsrnModel, RnanModel, MasaModel, JiifModel, LiifModel, SmsrModel, RcanModel, DrnModel, PhysicssrModel, DdbpnModel
from PIL import Image
import requests

name_to_class = {
    "a2n": A2nModel,
    "awsrn-bam": AwsrnModel,
    "carn": CarnModel,
    "carn-bam": CarnModel,
    "drln": DrlnModel,
    "drln-bam": DrlnModel,
    "edsr": EdsrModel,
    "edsr-base": EdsrModel,
    "msrn": MsrnModel,
    "mdsr": MdsrModel,
    "msrn-bam": MsrnModel,
    "mdsr-bam": MdsrModel,
    "pan": PanModel,
    "pan-bam": PanModel,
    "rcan-bam": RcanModel,
    "han": HanModel,
}

# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestEdsrConvertModel(TestTorchConvertModel):
    def load_model_impl(self, model_name, model_link):
        # image link from https://github.com/eugenesiow/super-image
        url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
        image = Image.open(requests.get(url, stream=True).raw)
        assert model_name in name_to_class, "Unexpected model name"
        model = name_to_class[model_name].from_pretrained(
            f'eugenesiow/{model_name}', scale=self.scale)
        inputs = ImageLoader.load_image(image)
        self.example = (torch.randn_like(inputs),)
        self.inputs = (inputs,)
        return model

    def teardown_method(self):
        # remove all downloaded files from cache
        cleanup_dir(hf_hub_cache_dir)
        super().teardown_method()

    @pytest.mark.parametrize("name,scale", [("edsr", 2)])
    @pytest.mark.precommit
    def test_convert_model_precommit(self, name, scale, ie_device):
        self.scale = scale
        self.run(name, None, ie_device)

    @pytest.mark.nightly
    @pytest.mark.parametrize("name,scale", [
        ("a2n", random.randint(2, 4)),
        ("awsrn-bam", random.randint(2, 4)),
        ("carn", random.randint(2, 4)),
        ("carn-bam", random.randint(2, 4)),
        ("drln", random.randint(2, 4)),
        ("drln-bam", random.randint(2, 4)),
        ("edsr", random.randint(2, 4)),
        ("edsr-base", random.randint(2, 4)),
        ("msrn", random.randint(2, 4)),
        ("msrn-bam", random.randint(2, 4)),
        ("mdsr", random.randint(2, 4)),
        ("mdsr-bam", random.randint(2, 4)),
        ("pan", random.randint(2, 4)),
        ("pan-bam", random.randint(2, 4)),
        ("han", 4),
        ("rcan-bam", 4),
    ])
    def test_convert_model_all_models(self, name, scale, ie_device):
        self.scale = scale
        self.run(name, None, ie_device)
