# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
from openvino._offline_transformations import stateful_to_stateless_transformation
from optimum.intel import OVModelForCausalLM
from models_hub_common.utils import retry
import models_hub_common.utils as utils
import pytest
import os
import re

def get_read_value_ops(model: ov.Model):
    return [op for op in model.get_ops() if op.get_type_name() == 'ReadValue']

def check_desc_tensors(expected_tensors, tensors):
    # checks if tensors descriptors are same as expected
    pattern_input = re.compile(R"input_restored.((past_key_values|present)\.(\d+)\.(key|value))")

    assert len(expected_tensors) == len(tensors)
    for expected in expected_tensors:
        # The `patch_stateful` in optimum use any name instead found key/value names OV will use names restore path
        # Restore expected names to find tensor for compare (can be removed when HG optimum updated)
        expected_names = {m[1] if m else name for m, name in ((pattern_input.match(name), name) for name in expected.names)}
        # tensor names check is relaxed the expected is sub-set of final names
        t_candidates = [t for t in tensors if expected_names.issubset(t.names)]
        assert len(t_candidates) == 1
        tensor = t_candidates[0]
        assert expected.get_element_type() == tensor.get_element_type()
        assert expected.get_partial_shape() == tensor.get_partial_shape()


def check_result_desc_tensors(expected_tensors, tensors):
    # checks if Result tensors descriptors are same as expected
    pattern_restore_output = re.compile(R"output_restored.((past_key_values|present)\.(\d+)\.(key|value))")
    pattern_output = re.compile(R"(present\.(\d+)\.(key|value))")

    assert len(expected_tensors) == len(tensors)
    for expected in expected_tensors:
        # The `patch_stateful` in optimum use any name instead found key/value names OV will use names restore path
        # Restore expected names to find tensor for compare (can be removed when HG optimum updated)
        expected_names = {name for name in expected.names if not pattern_restore_output.match(name)}
        expected_o_names = {name for name in expected.names if pattern_output.match(name)}
        expected_names = expected_o_names if expected_o_names else expected_names
        t_candidates = [t for t in tensors if expected_names.issubset(t.names)]
        assert len(t_candidates) == 1
        tensor = t_candidates[0]
        assert expected.get_element_type() == tensor.get_element_type()
        assert expected.get_partial_shape() == tensor.get_partial_shape()


@retry(3, exceptions=(OSError,), delay=1)
def run_stateful_to_stateless_in_runtime(tmp_path, model_id, model_link):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, stateful=True, compile=False)
    assert len(model.model.get_sinks()), f"Input model is not in the expected stateful form because it doesn't have any sinks."
    assert len(get_read_value_ops(model.model)), f"Input model is not in the expected stateful form because it doesn't have any ReadValue operations."

    stateful_to_stateless_transformation(model.model)

    sink_ops = model.model.get_sinks()
    read_value_ops = get_read_value_ops(model.model)
    assert len(sink_ops) == 0, f"Expected stateless model, but there are sinks found: {sink_ops}"
    assert len(read_value_ops) == 0, f"Expected stateless model, but there are ReadValue operations found: {read_value_ops}"

    stateless_model = OVModelForCausalLM.from_pretrained(model_id, export=True, stateful=False, compile=False)

    print(model.model)
    print(stateless_model.model)
    check_desc_tensors(model.model.inputs, stateless_model.model.inputs)
    check_result_desc_tensors(model.model.outputs, stateless_model.model.outputs)

    core = ov.Core()
    core.compile_model(model.model, 'CPU')


@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "tiny-set-stateful-models-precommit")))
def test_stateful_to_stateless_precommit(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_stateful_to_stateless_in_runtime(tmp_path, model_name, model_link)
