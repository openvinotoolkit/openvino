# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino._offline_transformations import paged_attention_transformation
from openvino._pyopenvino.op import _PagedAttentionExtension
from optimum.intel import OVModelForCausalLM
from optimum.intel.openvino import OVModelForVisualCausalLM
from typing import Type, Union
import openvino as ov
from models_hub_common.utils import retry
import models_hub_common.utils as utils
from sdpa2pa_ref_diff import ref_diff_map, ref_diff_map_cache_eviction, nodes_to_compare
import pytest
import os
import re

def compare_diffs(ov_model: ov.Model,
                  model_id: str,
                  use_block_indices_inputs: bool,
                  use_score_outputs: bool,
                  allow_cache_rotation: bool):
    before_map = {}
    for op in ov_model.get_ordered_ops():
        if op.get_type_name() in nodes_to_compare:
            before_map[op.get_type_name()] = before_map.get(op.get_type_name(), 0) + 1

    paged_attention_transformation(ov_model, use_block_indices_inputs, use_score_outputs, allow_cache_rotation)

    after_map = {}
    for op in ov_model.get_ordered_ops():
        if op.get_type_name() in nodes_to_compare:
            after_map[op.get_type_name()] = after_map.get(op.get_type_name(), 0) + 1

    # Collect the changes of nodes from nodes_to_compare
    # And check if the numbers correspond to the reference ones
    resulting_map = {}
    for op in set(after_map.keys()) | set(before_map.keys()):
        resulting_map[op] = after_map.get(op, 0) - before_map.get(op, 0)

    use_cache_eviction = use_block_indices_inputs and use_score_outputs and allow_cache_rotation
    reference_map = ref_diff_map_cache_eviction[model_id] if use_cache_eviction else ref_diff_map[model_id]

    assert reference_map == resulting_map

    model_inputs = ov_model.inputs
    for input in model_inputs:
        names = list(input.get_names()) # names stored in as set (in this case usually of 1 element)
        for name in names:
            if (("key_cache." in name) or ("value_cache." in name)):
                shape = input.get_partial_shape()
                # PagedAttention uses key_cache and value_cache inputs so the last 2 dimensions have to be static
                assert shape[-1].is_static, f"Dimension {len(shape) - 1} of input '{name}' in '{model_id}' is not static: {shape}"
                assert shape[-2].is_static, f"Dimension {len(shape) - 2} of input '{name}' in '{model_id}' is not static: {shape}"

    interesting_input_patterns = {}
    interesting_output_patterns = {}


    # Test for block_indices inputs and scores outputs to appear in the model
    if (use_block_indices_inputs):
        interesting_input_patterns["block_indices"] = r'^block_indices\.[0-9]+'

    if (use_score_outputs):
        interesting_output_patterns["scores"] = r'^scores\.[0-9]+'

    if (allow_cache_rotation):
        interesting_input_patterns["rotated_block_indices"] = r'^rotated_block_indices\.[0-9]+';
        interesting_input_patterns["rotation_deltas"] = r'^rotation_deltas\.[0-9]+';
        interesting_input_patterns["rotation_trig_lut"] = r'rotation_trig_lut';

    input_counters = {k: 0 for k in interesting_input_patterns}
    output_counters = {k: 0 for k in interesting_output_patterns}

    for pattern_dict, counter_dict, io_set in zip([interesting_input_patterns, interesting_output_patterns],
                                                  [input_counters, output_counters],
                                                [ov_model.inputs, ov_model.outputs]):
        for input_id in counter_dict:
            pattern = pattern_dict[input_id]
            for model_io in io_set:
                for name in list(model_io.get_names()):
                    if re.search(pattern, name):
                        counter_dict[input_id] += 1

    if allow_cache_rotation:
        assert input_counters["rotation_trig_lut"] == 1
        input_counters.pop("rotation_trig_lut")

    for input_id, count in input_counters.items():
        assert count == resulting_map["PagedAttentionExtension"], \
               f"The number of {input_id} inputs doesn't correspond to the expected value. Expected {resulting_map['PagedAttentionExtension']}, received {count}"

    for output_id, count in output_counters.items():
        assert count == resulting_map["PagedAttentionExtension"], \
               f"The number of {output_id} outputs doesn't correspond to the expected value. Expected {resulting_map['PagedAttentionExtension']}, received {count}"


@retry(3, exceptions=(OSError,), delay=1)
def run_pa(tmp_path,
           model_id,
           model_link,
           cls: Union[Type[OVModelForCausalLM], Type[OVModelForVisualCausalLM]],
           use_block_indices_inputs,
           use_score_outputs,
           allow_cache_rotation):
    model = cls.from_pretrained(model_id, export=True, trust_remote_code=True)
    ov_model = model.model if cls is OVModelForCausalLM else model.lm_model

    compare_diffs(ov_model, model_id, use_block_indices_inputs, use_score_outputs, allow_cache_rotation)

@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-models-precommit")))
def test_pa_precommit(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_pa(tmp_path, model_name, model_link, OVModelForCausalLM, False, False, False)

@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-models-precommit")))
def test_pa_precommit_use_cache_eviction(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_pa(tmp_path, model_name, model_link, OVModelForCausalLM, True, True, True)

@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-vl-models-precommit")))
def test_pa_vlm(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_pa(tmp_path, model_name, model_link, OVModelForVisualCausalLM, False, False, False)

@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-vl-models-precommit")))
def test_pa_vlm_use_cache_eviction(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_pa(tmp_path, model_name, model_link, OVModelForVisualCausalLM, True, True, True)
