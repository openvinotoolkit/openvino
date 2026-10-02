# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from huggingface_hub import snapshot_download
from openvino._offline_transformations import paged_attention_transformation
from openvino._pyopenvino import Type as OVType
from optimum.intel import OVModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
import openvino as ov
from models_hub_common.utils import retry
import models_hub_common.utils as utils
import pytest
import os
import platform

# Number of linear-attention layers each model is expected to fuse into PagedGatedDeltaNet.
ref_paged_gated_delta_net_count = {
    "optimum-intel-internal-testing/tiny-random-qwen3-next": 3,
}

STATE_TABLE_PREFIX = "gated_delta_state_table."

# Runtime inputs 6..10 of PagedGatedDeltaNet, in operation order.
RUNTIME_INPUT_NAMES = (
    "subsequence_begins",
    "la.block_indices",
    "la.block_indices_begins",
    "la.past_lens",
    "la.cache_interval",
)


def get_tensor_name(output: ov.Output) -> str:
    names = list(output.get_names())
    assert len(names) == 1, f"Expected a single tensor name, got {names}"
    return names[0]


def verify_paged_gated_delta_net(node, model_id: str):
    """
    Check a fused PagedGatedDeltaNet against the operation specification.

    query/key are [batch_size_in_tokens, num_heads, key_head_dim], value is
    [batch_size_in_tokens, v_num_heads, value_head_dim], the recurrent state table is
    [num_blocks, v_num_heads, value_head_dim, key_head_dim] and the output repeats the
    value shape. Grouped-query attention allows v_num_heads to be a multiple of num_heads,
    so the value-derived dimensions must not be taken from query.
    """
    query_shape = node.get_input_partial_shape(0)
    key_shape = node.get_input_partial_shape(1)
    value_shape = node.get_input_partial_shape(2)
    state_shape = node.get_input_partial_shape(3)

    for port, shape in ((0, query_shape), (1, key_shape), (2, value_shape)):
        assert shape.rank.get_length() == 3, \
            f"{model_id}: input {port} of {node.get_friendly_name()} must be 3D, got {shape}"

    num_heads = query_shape[1].get_length()
    key_head_dim = key_shape[2].get_length()
    v_num_heads = value_shape[1].get_length()
    value_head_dim = value_shape[2].get_length()
    assert v_num_heads % num_heads == 0, \
        f"{model_id}: v_num_heads {v_num_heads} is not a multiple of num_heads {num_heads}"

    state_table = node.input_value(3).get_node()
    assert state_table.get_type_name() == "Parameter", \
        f"{model_id}: recurrent_state_table must be a Parameter, got {state_table.get_type_name()}"
    state_table_name = get_tensor_name(node.input_value(3))
    assert state_table_name.startswith(STATE_TABLE_PREFIX), \
        f"{model_id}: unexpected recurrent_state_table name '{state_table_name}'"
    assert state_shape.rank.get_length() == 4 and state_shape[0].is_dynamic, state_shape
    assert [state_shape[i].get_length() for i in (1, 2, 3)] == [v_num_heads, value_head_dim, key_head_dim], \
        f"{model_id}: recurrent_state_table {state_shape} does not match " \
        f"[num_blocks, {v_num_heads}, {value_head_dim}, {key_head_dim}]"

    for port, name in enumerate(RUNTIME_INPUT_NAMES, start=6):
        assert get_tensor_name(node.input_value(port)) == name, \
            f"{model_id}: input {port} of {node.get_friendly_name()} is " \
            f"'{get_tensor_name(node.input_value(port))}', expected '{name}'"
        assert node.get_input_element_type(port) == OVType.i32, \
            f"{model_id}: input {port} of {node.get_friendly_name()} must be i32"

    assert node.get_output_partial_shape(0) == value_shape, \
        f"{model_id}: output {node.get_output_partial_shape(0)} must repeat the value shape {value_shape}"

    # The fusion restores the original GatedDeltaNet layout [batch, seq_len, v_num_heads, value_head_dim].
    consumers = list(node.output(0).get_target_inputs())
    assert len(consumers) == 1, f"{model_id}: expected a single consumer of {node.get_friendly_name()}"
    restored = consumers[0].get_node()
    assert restored.get_type_name() == "Reshape", \
        f"{model_id}: expected the output to be reshaped back, got {restored.get_type_name()}"
    restored_shape = restored.get_output_partial_shape(0)
    assert restored_shape.rank.get_length() == 4, restored_shape
    assert [restored_shape[2].get_length(), restored_shape[3].get_length()] == [v_num_heads, value_head_dim], \
        f"{model_id}: restored output {restored_shape} must keep [{v_num_heads}, {value_head_dim}]"

    return state_table_name


def apply_transformation_and_verify(ov_model: ov.Model, model_id: str, ie_device: str, expected_count: int = None):
    if expected_count is None:
        expected_count = ref_paged_gated_delta_net_count[model_id]

    # SDPAToPagedAttention first folds the exported linear-attention subgraphs into
    # GatedDeltaNet, then PagedGatedDeltaNetFusion rewrites them for paged execution.
    paged_attention_transformation(ov_model, False, False, False, False, False, False, False)

    assert not any(op.get_type_name() == "GatedDeltaNet" for op in ov_model.get_ordered_ops()), \
        f"{model_id}: GatedDeltaNet is left unfused"

    state_table_names = [verify_paged_gated_delta_net(op, model_id)
                         for op in ov_model.get_ordered_ops() if op.get_type_name() == "PagedGatedDeltaNet"]
    assert len(state_table_names) == expected_count, \
        f"{model_id}: expected {expected_count} PagedGatedDeltaNet operations, got {len(state_table_names)}"
    assert len(set(state_table_names)) == expected_count, \
        f"{model_id}: recurrent state tables are shared between layers: {state_table_names}"

    model_input_names = {name for model_input in ov_model.inputs for name in model_input.get_names()}
    for name in (*RUNTIME_INPUT_NAMES, *state_table_names):
        assert name in model_input_names, f"{model_id}: '{name}' is not exposed as a model input"

    ov.Core().compile_model(ov_model, ie_device)


@retry(3, exceptions=(OSError,), delay=1)
def run_gated_delta_net_pa(model_id, ie_device):
    model_cached = snapshot_download(model_id)  # required to avoid HF rate limits
    model = OVModelForCausalLM.from_pretrained(model_cached, export=True, trust_remote_code=True, compile=False)
    apply_transformation_and_verify(model.model, model_id, ie_device)


@retry(3, exceptions=(OSError,), delay=1)
def run_gated_delta_net_pa_grouped_query(tmp_path, model_id, num_groups, ie_device):
    """
    Exercise grouped-query linear attention, where v_num_heads is a multiple of num_heads.

    The tiny reference model uses one value head per key head, so the grouped case is only
    reachable through a synthetic config with a single linear-attention layer.
    """
    config = AutoConfig.from_pretrained(snapshot_download(model_id))
    config.linear_num_value_heads = config.linear_num_key_heads * num_groups
    config.layer_types = ["linear_attention", "full_attention"]
    config.num_hidden_layers = len(config.layer_types)

    model_path = os.path.join(tmp_path, f"synthetic_gqa_{num_groups}")
    AutoModelForCausalLM.from_config(config).save_pretrained(model_path)

    model = OVModelForCausalLM.from_pretrained(model_path, export=True, trust_remote_code=True, compile=False)
    apply_transformation_and_verify(model.model, model_id, ie_device, expected_count=1)


GDN_PRECOMMIT_TEST_CASES = utils.get_models_list(
    os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-gdn-models-precommit")
)


@pytest.mark.precommit
@pytest.mark.parametrize("model_info_tuple", GDN_PRECOMMIT_TEST_CASES, ids=lambda entry: entry[0])
def test_gated_delta_net_pa_precommit(model_info_tuple, ie_device):
    model_name, model_link, mark, reason = model_info_tuple
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if platform.machine() in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64']:
        pytest.skip("PagedAttention tests are not enabled on ARM")
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)

    run_gated_delta_net_pa(model_name, ie_device)


@pytest.mark.precommit
@pytest.mark.parametrize("model_info_tuple", GDN_PRECOMMIT_TEST_CASES, ids=lambda entry: entry[0])
@pytest.mark.parametrize("num_groups", [2, 4])
def test_gated_delta_net_pa_grouped_query_precommit(tmp_path, model_info_tuple, num_groups, ie_device):
    model_name, model_link, mark, reason = model_info_tuple
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if platform.machine() in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64']:
        pytest.skip("PagedAttention tests are not enabled on ARM")
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)

    run_gated_delta_net_pa_grouped_query(tmp_path, model_name, num_groups, ie_device)
