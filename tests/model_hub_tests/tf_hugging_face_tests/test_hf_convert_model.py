# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from models_hub_common.test_convert_model import TestConvertModel
from transformers import TFAutoModel
from models_hub_common.utils import get_models_list
import os

class TestTFHuggingFace(TestConvertModel):
    def load_model(self, model_name, model_link):
        return TFAutoModel.from_pretrained(model_name)

    def get_inputs_info(self, model):
        self.example_input = model.dummy_inputs
        return None

    def infer_fw_model(self, model_obj, inputs):
        outputs = model_obj(inputs)
        if isinstance(outputs, dict):
            post_outputs = {}
            for out_name, out_value in outputs.items():
                post_outputs[out_name] = out_value.numpy()
        elif isinstance(outputs, list):
            post_outputs = []
            for out_value in outputs:
                post_outputs.append(out_value.numpy())
        else:
            post_outputs = [outputs.numpy()]

        return post_outputs

    def prepare_inputs(self, inputs):
        return self.example_input


    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             get_models_list(os.path.join(os.path.dirname(__file__), "precommit_models")))
    @pytest.mark.precommit
    def test_tf_hugging_face_precommit(self, model_name, model_link, mark, reason, ie_device):
        self.run(model_name, '', ie_device)


    @pytest.mark.parametrize("model_name,model_link,mark,reason",
                             get_models_list(os.path.join(os.path.dirname(__file__), "nightly_models")))
    @pytest.mark.nightly
    def test_tf_hugging_face_nightly(self, model_name, model_link, mark, reason, ie_device):
        self.run(model_name, '', ie_device)