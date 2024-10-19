# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino._offline_transformations import paged_attention_transformation
from openvino._pyopenvino.op import _PagedAttentionExtension
from optimum.intel import OVModelForCausalLM
from models_hub_common.utils import retry
import models_hub_common.utils as utils
from sdpa2pa_ref_diff import ref_diff_map, ref_diff_map_cache_eviction, nodes_to_compare
import pytest
import os
import re

@retry(3, exceptions=(OSError,), delay=1)
def run_pa(tmp_path, model_id, model_link, use_block_indices_inputs, use_score_outputs):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)

    before_map = {}
    for op in model.model.get_ordered_ops():
        if op.get_type_name() in nodes_to_compare:
            before_map[op.get_type_name()] = before_map.get(op.get_type_name(), 0) + 1

    paged_attention_transformation(model.model, use_block_indices_inputs, use_score_outputs)

    after_map = {}
    for op in model.model.get_ordered_ops():
        if op.get_type_name() in nodes_to_compare:
            after_map[op.get_type_name()] = after_map.get(op.get_type_name(), 0) + 1

    # Collect the changes of nodes from nodes_to_compare
    # And check if the numbers correspond to the reference ones
    resulting_map = {}
    for op in set(after_map.keys()) | set(before_map.keys()):
        resulting_map[op] = after_map.get(op, 0) - before_map.get(op, 0)

    use_cache_eviction = use_block_indices_inputs and use_score_outputs
    reference_map = ref_diff_map_cache_eviction[model_id] if use_cache_eviction else ref_diff_map[model_id]

    assert reference_map == resulting_map

    model_inputs = model.model.inputs
    for input in model_inputs:
        names = list(input.get_names()) # names stored in as set (in this case usually of 1 element)
        for name in names:
            if (("key_cache." in name) or ("value_cache." in name)):
                shape = input.get_partial_shape()
                # PagedAttention uses key_cache and value_cache inputs so the last 2 dimensions have to be static
                assert shape[-1].is_static, f"Dimension {len(shape) - 1} of input '{name}' in '{model_id}' is not static: {shape}"
                assert shape[-2].is_static, f"Dimension {len(shape) - 2} of input '{name}' in '{model_id}' is not static: {shape}"

    # Test for block_indices inputs and scores outputs to appear in the model 
    if (use_block_indices_inputs):
        block_indices_pattern = r'block_indices\.[0-9]+'
        block_indices_counter = 0

        model_inputs = model.model.inputs
        for input in model_inputs:
            for name in list(input.get_names()):
                if re.search(block_indices_pattern, name):
                    block_indices_counter += 1

        assert block_indices_counter == resulting_map["PagedAttentionExtension"], \
               f"The number of block_indices inputs doesn't correspond to the expected value. Expected {resulting_map['PagedAttentionExtension']}, received {block_indices_counter}"
    
    if (use_score_outputs):
        score_pattern = r'scores\.[0-9]+'
        score_outputs_counter = 0

        model_outputs = model.model.outputs
        for output in model_outputs:
            for name in list(output.get_names()):
                if re.search(score_pattern, name):
                    score_outputs_counter += 1

        assert block_indices_counter == resulting_map["PagedAttentionExtension"], \
               f"The number of scores outputs doesn't correspond to the expected value. Expected {resulting_map['PagedAttentionExtension']}, received {block_indices_counter}"

@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-models-precommit")))
def test_pa_precommit(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_pa(tmp_path, model_name, model_link, False, False)

@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-models-precommit")))
def test_pa_precommit_use_cache_eviction(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_pa(tmp_path, model_name, model_link, True, True)