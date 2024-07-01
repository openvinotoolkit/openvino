# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov
from openvino._offline_transformations import stateful_to_stateless_transformation
from optimum.intel import OVModelForCausalLM
import models_hub_common.utils as utils
import pytest
import os

def get_read_value_ops(model: ov.Model):
    return [op for op in model.get_ops() if op.get_type_name() == 'ReadValue']

def run_stateful_to_stateless_in_runtime(tmp_path, model_id, model_link):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, stateful=True, compile=False)
    assert len(model.model.get_sinks()), f"Input model is not in the expected stateful form because it doesn't have any sinks."
    assert len(get_read_value_ops(model.model)), f"Input model is not in the expected stateful form because it doesn't have any ReadValue operations."

    stateful_to_stateless_transformation(model.model)

    sink_ops = model.model.get_sinks()
    read_value_ops = get_read_value_ops(model.model)
    assert len(sink_ops) == 0, f"Expected stateless model, but there are sinks found: {sink_ops}"
    assert len(read_value_ops) == 0, f"Expected stateless model, but there are ReadValue operations found: {read_value_ops}"
    # TODO: check names of inputs/output, their count and topological equivalence in respect to the entire model
    # TODO: compare with originally stateless model


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