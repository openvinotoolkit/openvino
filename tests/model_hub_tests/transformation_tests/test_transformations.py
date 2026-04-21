# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from huggingface_hub import snapshot_download
import models_hub_common.utils as utils
import pytest
import os
import platform
import openvino as ov
import tempfile
from collections import deque
import csv
import torch


def parse_transformations_log(file_name):
    with open(file_name, 'r') as f_in:
        csv_reader = csv.reader(f_in, delimiter=';')
        for line in csv_reader:
            if line[0] != 't':
                continue
            ts_name = line[1]
            status = line[4]
            yield ts_name, status


def check_transformations(file_name, ts_names):
    not_executed = deque(ts_names)
    false_executed = set()
    for ts_name, status in parse_transformations_log(file_name):
        if not not_executed and not false_executed:
            break
        for _ in range(len(not_executed)):
            not_executed_name = not_executed.popleft()
            if not_executed_name == ts_name:
                if status != '1':
                    false_executed.add(not_executed_name)
                break
            not_executed.append(not_executed_name)
        if ts_name in false_executed and status == '1':
            false_executed.remove(ts_name)
    if not_executed or false_executed:
        fail_text = ''
        if not_executed:
            not_executed_names = ','.join(not_executed)
            fail_text = f'transformation(s) {not_executed_names} not executed'
        if false_executed:
            false_executed_names = ','.join(false_executed)
            if bool(fail_text):
                fail_text += '; '
            fail_text += f'transformation(s) {false_executed_names} executed with false return'
        pytest.fail(fail_text)


def check_operations(actual_layer_types, expected_layer_types):
    not_found = [layer for layer in expected_layer_types if layer not in actual_layer_types]
    if not_found:
        names = ','.join(not_found)
        pytest.fail(f'operation(s) {names} not found in compiled model')


class EnvVar:
    def __init__(self, env_vars):
        self.__vars = env_vars

    def __enter__(self):
        for name, value in self.__vars.items():
            os.environ[name] = value

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name in self.__vars:
            del os.environ[name]


def compile_and_check(ov_model, ie_device, ts_names, expected_layer_types):
    with tempfile.NamedTemporaryFile(delete=True) as temp_file, \
            EnvVar({'OV_ENABLE_PROFILE_PASS': temp_file.name}):
        core = ov.Core()
        compiled = core.compile_model(ov_model, ie_device)
        check_transformations(temp_file.name, ts_names)
        runtime_model = compiled.get_runtime_model()
        type_names = {op.get_rt_info()["layerType"] for op in runtime_model.get_ordered_ops()}
        check_operations(type_names, expected_layer_types)


def run_test(model_id, ie_device, ts_names, expected_layer_types):
    try:
        from optimum.intel import OVModelForCausalLM
    except (ImportError, AttributeError):
        try:
            from optimum.intel.openvino import OVModelForCausalLM
        except (ImportError, AttributeError):
            pytest.skip("OVModelForCausalLM unavailable with installed package versions")

    model_cached = snapshot_download(model_id)  # required to avoid HF rate limits
    try:
        model = OVModelForCausalLM.from_pretrained(model_cached, export=True, trust_remote_code=True)
    except (ValueError, ImportError) as e:
        pytest.skip(f"model export is not possible with the installed package versions: {e}")

    compile_and_check(model.model, ie_device, ts_names, expected_layer_types)


def run_flux_test(model_id, ie_device, ts_names, expected_layer_types):
    from diffusers import FluxTransformer2DModel

    model_cached = snapshot_download(model_id)
    try:
        transformer = FluxTransformer2DModel.from_pretrained(
            model_cached, subfolder="transformer")
    except Exception as e:
        pytest.skip(f"flux model loading failed: {e}")

    transformer.eval()
    config = transformer.config

    batch_size = 1
    height, width = 8, 8
    image_seq_len = height * width
    text_seq_len = 4
    in_channels = config.in_channels
    joint_attention_dim = config.joint_attention_dim
    pooled_projection_dim = config.pooled_projection_dim

    hidden_states = torch.randn(batch_size, image_seq_len, in_channels)
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, joint_attention_dim)
    pooled_projections = torch.randn(batch_size, pooled_projection_dim)
    timestep = torch.tensor([500.0])
    img_ids = torch.zeros(image_seq_len, 3)
    for h in range(height):
        for w in range(width):
            img_ids[h * width + w, 1] = h
            img_ids[h * width + w, 2] = w
    txt_ids = torch.zeros(text_seq_len, 3)

    example_input = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_projections": pooled_projections,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
    }

    try:
        ov_model = ov.convert_model(transformer, example_input=example_input)
    except Exception as e:
        pytest.skip(f"flux model conversion to OpenVINO failed: {e}")

    compile_and_check(ov_model, ie_device, ts_names, expected_layer_types)


@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason, ts_names, layer_types, model_type", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "transformations-models-precommit")))
def test_transformations_precommit(tmp_path, model_name, model_link, mark, reason, ie_device, ts_names, layer_types, model_type):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    arm_machine_names = {'arm', 'armv7l', 'aarch64', 'arm64', 'ARM64'}
    arm_unsupported_rope_models = {
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    }
    # RoPEFusionFlux and RoPEFusionLtxVideo are disabled on CPU
    gpu_only_rope_models = {
        "hf-internal-testing/tiny-flux-pipe",
    }
    # RoPEFusionGPTJ and RoPEFusionIOSlicing are disabled on GPU
    gpu_disabled_rope_models = {
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        "hf-internal-testing/tiny-random-CodeGenForCausalLM",
    }
    if platform.machine() in arm_machine_names and model_name in arm_unsupported_rope_models:
        pytest.skip("RoPE fusion is not available for this model on ARM")
    if model_name in gpu_only_rope_models and ie_device == "CPU":
        pytest.skip("RoPE fusion for this model is disabled on CPU")
    if model_name in gpu_disabled_rope_models and ie_device == "GPU":
        pytest.skip("RoPE fusion for this model is disabled on GPU")
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    if not ts_names and not layer_types:
        return
    if model_type == 'flux':
        run_flux_test(model_name, ie_device, ts_names, layer_types)
    else:
        run_test(model_name, ie_device, ts_names, layer_types)
