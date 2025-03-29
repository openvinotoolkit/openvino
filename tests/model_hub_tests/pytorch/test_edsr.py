# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import random
import torch
from models_hub_common.constants import hf_cache_dir, clean_hf_cache_dir
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
    def load_model(self, model_name, model_link):
        # image link from https://github.com/eugenesiow/super-image
        url = 'https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg'
        image = Image.open(requests.get(url, stream=True).raw)
        assert model_name in name_to_class, "Unexpected model name"
        print(f"scale: {self.scale}")
        model = name_to_class[model_name].from_pretrained(
            f'eugenesiow/{model_name}', scale=self.scale)
        inputs = ImageLoader.load_image(image)
        self.example = (torch.randn_like(inputs),)
        self.inputs = (inputs,)
        return model

    def teardown_method(self):
        if clean_hf_cache_dir:
            # remove all downloaded files from cache
            cleanup_dir(hf_cache_dir)
        super().teardown_method()

    @pytest.mark.parametrize("name", ["edsr"])
    @pytest.mark.precommit
    def test_convert_model_precommit(self, name, ie_device):
        self.scale = random.randint(2, 4)
        self.run(name, None, ie_device)

    @pytest.mark.nightly
    @pytest.mark.parametrize("name", [
        "a2n",
        "awsrn-bam",
        "carn",
        "carn-bam",
        "drln",
        "drln-bam",
        "edsr",
        "edsr-base",
        "msrn",
        "msrn-bam",
        "mdsr",
        "mdsr-bam",
        "pan",
        "pan-bam",
        "han",
        "rcan-bam",
    ])
    def test_convert_model_all_models(self, name, ie_device):
        if name in ["han", "rcan-bam"]:
            self.scale = 4
        else:
            self.scale = random.randint(2, 4)
        self.run(name, None, ie_device)
