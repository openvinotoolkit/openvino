# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from optimum.intel import OVModelForCausalLM
import models_hub_common.utils as utils
import pytest
import os
import openvino as ov
import tempfile
from collections import deque
import csv


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


def run_test(model_id, ie_device, ts_names, expected_layer_types):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(delete=True) as temp_file, \
            EnvVar({'OV_ENABLE_PROFILE_PASS': temp_file.name}):
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
