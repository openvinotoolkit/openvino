# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
from openvino._offline_transformations import stateful_to_stateless_transformation
from optimum.intel import OVModelForCausalLM
from models_hub_common.utils import retry
import models_hub_common.utils as utils
import pytest
import os

def get_read_value_ops(model: ov.Model):
    return [op for op in model.get_ops() if op.get_type_name() == 'ReadValue']

def check_desc_tensors(tensors1, tensors2):
    # order of tensors may not match, comparing by the total amount and names
    assert len(tensors1) == len(tensors2)
    assert set(tuple(t.names) for t in tensors1) == set(tuple(t.names) for t in tensors2)
    for t1 in tensors1:
        t2_candidates = [t for t in tensors2 if t1.names & t.names]
        assert len(t2_candidates) == 1
        t2 = t2_candidates[0]
        assert t1.names == t2.names
        assert t1.get_partial_shape() == t2.get_partial_shape()
        assert t1.get_element_type() == t2.get_element_type()

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
    check_desc_tensors(model.model.outputs, stateless_model.model.outputs)

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