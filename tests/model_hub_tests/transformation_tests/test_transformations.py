# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from optimum.intel import OVModelForCausalLM
import models_hub_common.utils as utils
import pytest
import os
import openvino as ov
import tempfile
from collections import deque


def check_transformations(file_name, ts_names):
    ts_names_dq = deque((';{};'.format(name) for name in ts_names))
    with open(file_name, 'r') as f_in:
        for line in f_in:
            if not ts_names_dq:
                break
            for _ in range(len(ts_names_dq)):
                name = ts_names_dq.popleft()
                if name in line:
                    break
                ts_names_dq.append(name)
    if ts_names_dq:
        names = ','.join(s.replace(';', '') for s in ts_names_dq)
        pytest.fail(f'transformation(s) {names} not executed')


def check_operations(actual_layer_types, expected_layer_types):
    not_found = [layer for layer in expected_layer_types if layer not in actual_layer_types]
    if not_found:
        names = ','.join(not_found)
        pytest.fail(f'operation(s) {names} not found in compiled model')


def run_test(model_id, ie_device, ts_names, expected_layer_types):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        os.environ['OV_ENABLE_PROFILE_PASS'] = temp_file.name
        core = ov.Core()
        compiled = core.compile_model(model.model, ie_device)
        check_transformations(temp_file.name, ts_names)
        ov_model = compiled.get_runtime_model()
        type_names = {op.get_rt_info()["layerType"] for op in ov_model.get_ordered_ops()}
        check_operations(type_names, expected_layer_types)


@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason, ts_names, layer_types", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "transformations-models-precommit")))
def test_transformations_precommit(tmp_path, model_name, model_link, mark, reason, ie_device, ts_names, layer_types):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    if not ts_names and not layer_types:
        return
    run_test(model_name, ie_device, ts_names, layer_types)
