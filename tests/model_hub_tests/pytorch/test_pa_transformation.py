# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino._offline_transformations import paged_attention_transformation
from openvino._pyopenvino.op import _PagedAttentionExtension
from optimum.intel import OVModelForCausalLM
import models_hub_common.utils as utils
import openvino as ov
import pytest
import os

# def get_models_list(file_name: str):
#     models = []
#     with open(file_name) as f:
#         for model_name in f:
#             model_name = model_name.strip()
#             # skip comment in model scope file
#             if model_name.startswith('#'):
#                 continue
#             models.append(model_name)
#     return models

def run_pa(tmp_path, model_id):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True)

    # Uncomment if you'd like to save the post-Optimum IR
    # model_path = os.path.join(tmp_path, model_id)
    # model.save_pretrained(model_path)

    paged_attention_transformation(model.model)

    # Test that a _PagedAttentionExtension node appeared after the transformation.
    assert any(isinstance(op, _PagedAttentionExtension) for op in model.model.get_ordered_ops()), f"The model '{model_id}' has no _PagedAttentionExtension present."

    model_inputs = model.model.inputs
    for input in model_inputs:
        names = list(input.get_names()) # names stored in as set (in this case usually of 1 element)
        for name in names:
            if (("key_cache." in name) or ("value_cache." in name)):
                shape = input.get_partial_shape()
                assert shape[-1].is_static, f"Dimension {len(shape) - 1} of input '{name}' in '{model_id}' is not static: {shape}"
                assert shape[-2].is_static, f"Dimension {len(shape) - 2} of input '{name}' in '{model_id}' is not static: {shape}"

@pytest.mark.precommit
@pytest.mark.parametrize("model_id", utils.get_skipped_model_links("pytorch/models/hf-tiny-random-models-precommit"))
def test_pa_precommit(tmp_path, model_id, ie_device):
    run_pa(tmp_path, model_id)

@pytest.mark.nightly
@pytest.mark.parametrize("model_id", utils.get_skipped_model_links("pytorch/models/hf-tiny-random-models-nightly"))
def test_pa_nightly(tmp_path, model_id, ie_device):
    run_pa(tmp_path, model_id)