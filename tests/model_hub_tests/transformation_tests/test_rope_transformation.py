# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from optimum.intel import OVModelForCausalLM
import models_hub_common.utils as utils
import pytest
import os
import openvino as ov
import tempfile


def traverse_graph_recursive(node, visited, node_types):
    node_id = node.get_friendly_name()
    if node_id in visited:
        return
    visited.add(node_id)

    node_types.add(node.get_type_name())

    for node_input in node.input_values():
        traverse_graph_recursive(node_input.get_node(), visited, node_types)


def traverse_graph(model_outputs):
    node_types = set()
    visited = set()
    for model_output in model_outputs:
        traverse_graph_recursive(model_output.get_node(), visited, node_types)
    return node_types


def run_test(model_id):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        os.environ['OV_ENABLE_PROFILE_PASS'] = temp_file.name
        core = ov.Core()
        core.compile_model(model.model, 'CPU')
        has_rope_fusion = False
        with open(temp_file.name, 'r') as f_in:
            for line in f_in:
                if ';ov::pass::RoPEFusion;' in line:
                    has_rope_fusion = True
                    break
        if not has_rope_fusion:
            pytest.fail('no ov::pass::RoPEFusion called')


@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "rope-models-precommit")))
def test_rope_precommit(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_test(model_name)
