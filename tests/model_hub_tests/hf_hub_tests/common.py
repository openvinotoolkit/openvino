# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from optimum.intel import OVModelForCausalLM
from pathlib import Path
from openvino._pyopenvino.op import _PagedAttentionExtension
from openvino._offline_transformations import paged_attention_transformation
import openvino as ov

def get_models_list(file_name: str):
   models = []
   with open(file_name) as f:
       for model_name in f:
           model_name = model_name.strip()
           # skip comment in model scope file
           if model_name.startswith('#'):
               continue
           models.append(model_name)
   return models

def run_pa(tmp_path, model_id):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True)
    model_path : Path = tmp_path / model_id
    model.save_pretrained(model_path)

    core = ov.Core()
    ov_model = core.read_model(model=model_path / "openvino_model.xml")
    core.compile_model(ov_model) # Do I have to compile model?

    paged_attention_transformation(ov_model)
    # Checking just after SDPAToPA

    assert any(isinstance(op, _PagedAttentionExtension) for op in ov_model.get_ordered_ops()), f"{model_id} : no _PagedAttentionExtension present in the model."

    model_inputs = ov_model.inputs
    for input in model_inputs:
        names = list(input.get_names()) # names stored in as set (in this case usually of 1 element)
        for name in names:
            if (("key_cache." in name) or ("value_cache." in name)):
                # print(input.get_partial_shape().is_static)
                assert input.get_partial_shape().is_static, f"{model_id} {name} is not static.c."