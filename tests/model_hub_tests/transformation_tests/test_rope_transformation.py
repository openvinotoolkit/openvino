# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from optimum.intel import OVModelForCausalLM
import models_hub_common.utils as utils
import pytest
import os
import openvino as ov
import tempfile


def run_test(model_id, ie_device):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)

    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        os.environ['OV_ENABLE_PROFILE_PASS'] = temp_file.name
        core = ov.Core()
        compiled = core.compile_model(model.model, ie_device)
        has_rope_fusion = False
        with open(temp_file.name, 'r') as f_in:
            for line in f_in:
                if ';ov::pass::RoPEFusion;' in line:
                    has_rope_fusion = True
                    break
        if not has_rope_fusion:
            pytest.fail('ov::pass::RoPEFusion was not executed')
        ov_model = compiled.get_runtime_model()
        type_names = (op.get_rt_info()["layerType"] for op in ov_model.get_ordered_ops())
        if 'RoPE' not in type_names:
            pytest.fail('RoPE operation not found in compiled model')


@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "rope-models-precommit")))
def test_rope_precommit(tmp_path, model_name, model_link, mark, reason, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_test(model_name, ie_device)
