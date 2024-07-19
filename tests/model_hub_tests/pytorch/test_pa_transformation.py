# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino._offline_transformations import paged_attention_transformation
from openvino._pyopenvino.op import _PagedAttentionExtension
from optimum.intel import OVModelForCausalLM
import models_hub_common.utils as utils
import pytest
import os
import re

def run_pa(tmp_path, model_id, model_link, use_cache_eviction):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)

    paged_attention_transformation(model.model, use_cache_eviction)

    # Test that a _PagedAttentionExtension node appeared after the transformation.
    pa_counter = 0
    for op in model.model.get_ordered_ops():
        if isinstance(op, _PagedAttentionExtension):
            pa_counter += 1

    assert pa_counter > 0, f"The model '{model_id}' has no _PagedAttentionExtension present."

    model_inputs = model.model.inputs
    for input in model_inputs:
        names = list(input.get_names()) # names stored in as set (in this case usually of 1 element)
        for name in names:
            if (("key_cache." in name) or ("value_cache." in name)):
                shape = input.get_partial_shape()
                # PagedAttention uses key_cache and value_cache inputs so the last 2 dimensions have to be static
                assert shape[-1].is_static, f"Dimension {len(shape) - 1} of input '{name}' in '{model_id}' is not static: {shape}"
                assert shape[-2].is_static, f"Dimension {len(shape) - 2} of input '{name}' in '{model_id}' is not static: {shape}"

    model_inputs = model.model.inputs
    model_outputs = model.model.outputs

    # Test for block_indices inputs and scores outputs to appear in the model 
    block_indices_pattern = r'block_indices'
    block_indices_counter = 0

    score_pattern = r'scores\.[0-9]+'
    score_outputs_counter = 0

    if (use_cache_eviction):
        model_inputs = model.model.inputs
        for input in model_inputs:
            for name in list(input.get_names()):
                if block_indices_pattern == name:
                    block_indices_counter += 1

        assert(block_indices_counter == pa_counter)

        model_outputs = model.model.outputs
        for output in model_outputs:
            for name in list(output.get_names()):
                if re.search(score_pattern, name):
                    score_outputs_counter += 1

        assert(score_outputs_counter == pa_counter)

@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-models-precommit")))
def test_pa_precommit(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_pa(tmp_path, model_name, model_link, False)

@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-models-precommit")))
def test_pa_precommit_use_cache_eviction(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_pa(tmp_path, model_name, model_link, True)