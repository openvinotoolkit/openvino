# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
import os
import pytest
import requests
from PIL import Image
from models_hub_common.constants import hf_cache_dir, clean_hf_cache_dir
from models_hub_common.utils import cleanup_dir, get_models_list, retry
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    FlaxAutoModel,
    AutoImageProcessor
)
from transformers.image_processing_base import BatchFeature

from jax_utils import TestJaxConvertModel


class TestTransformersModel(TestJaxConvertModel):
    @retry(3, exceptions=(OSError,), delay=1)
    def load_model(self, model_name, _):
        model = FlaxAutoModel.from_pretrained(model_name)
        if model_name in ['google/vit-base-patch16-224-in21k']:
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)
            image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.example = image_processor(images=image, return_tensors="np")
        elif model_name in ['albert/albert-base-v2', 'facebook/bart-base', 'ksmcg/Mistral-tiny']:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.example = tokenizer("Hello, my dog is cute", return_tensors="np")
        elif model_name in ['openai/clip-vit-base-patch32']:
            processor = AutoProcessor.from_pretrained(model_name)
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)
            self.example = processor(text=["a photo of a cat", "a photo of a dog"],
                                     images=image, return_tensors="np", padding=True)
        if isinstance(self.example, BatchFeature):
            self.example = dict(self.example)
        return model

    def teardown_method(self):
        if clean_hf_cache_dir:
            # remove all downloaded files from cache
            cleanup_dir(hf_cache_dir)
        super().teardown_method()

    def infer_ov_model(self, ov_model, inputs, ie_device):
        # TODO: support original input tensor names
        if isinstance(inputs, dict):
            new_inputs = []
            for _, value in inputs.items():
                new_inputs.append(value)
            inputs = new_inputs
        compiled = ov.compile_model(ov_model, ie_device, self.ov_config)
        ov_outputs = compiled(inputs)
        return ov_outputs

    @pytest.mark.parametrize("type,name,mark,reason",
                             get_models_list(os.path.join(os.path.dirname(__file__), "hf_transformers_models")))
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_convert_model_all_models(self, name, type, mark, reason, ie_device):
        valid_marks = ['skip', 'xfail']
        assert mark is None or mark in valid_marks, f"Invalid case for {name}"
        if mark == 'skip':
            pytest.skip(reason)
        elif mark == 'xfail':
            pytest.xfail(reason)
        self.run(model_name=name, model_link=None, ie_device=ie_device)
