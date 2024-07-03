# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino._offline_transformations import paged_attention_transformation
from openvino._offline_transformations import dump_model
from openvino._pyopenvino.op import _PagedAttentionExtension
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer, pipeline
import models_hub_common.utils as utils
import pytest
import os
import openvino as ov

from openvino.runtime.op import Result

def run_pa(tmp_path, model_id, model_link):
    model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True, compile=False)
    # model = OVModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)

    # dump_model(model.model, "old_jais_before_pa")
    paged_attention_transformation(model.model)
    # dump_model(model.model, "old_jais_after_pa")

    # model.compile()

    # tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # inputs = tokenizer("Who is Jesus?", return_tensors="pt", return_token_type_ids=False)
    # gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
    # res = tokenizer.batch_decode(gen_tokens)
    # print(res)
    # assert len(res) == 1

    # core = ov.Core()
    # compiled_model = core.compile_model(model.model)
    # req = compiled_model.create_infer_request()
    # print(req)
    # print(req.get_tensor("ANDREW").data)
    # for op in model.model.get_ordered_ops():
    #     if (type(op) is Result):
    #         print(op)


    # print("---")
    # print(model.model.outputs)
    # print("---")
    # print(model.request)
    # print(model.request.get_tensor("ANDREW").data)

    # Test that a _PagedAttentionExtension node appeared after the transformation.
    # assert any(isinstance(op, _PagedAttentionExtension) for op in model.model.get_ordered_ops()), f"The model '{model_id}' has no _PagedAttentionExtension present."

    # model_inputs = model.model.inputs
    # for input in model_inputs:
    #     names = list(input.get_names()) # names stored in as set (in this case usually of 1 element)
    #     for name in names:
    #         if (("key_cache." in name) or ("value_cache." in name)):
    #             shape = input.get_partial_shape()
    #             # PagedAttention uses key_cache and value_cache inputs so the last 2 dimensions have to be static
    #             assert shape[-1].is_static, f"Dimension {len(shape) - 1} of input '{name}' in '{model_id}' is not static: {shape}"
    #             assert shape[-2].is_static, f"Dimension {len(shape) - 2} of input '{name}' in '{model_id}' is not static: {shape}"

@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-models-precommit")))
def test_pa_precommit(tmp_path, model_name, model_link, mark, reason, ie_device, capsys):
    if (ie_device == 'GPU'):
        pytest.skip("SKIPPING GPU")
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_pa(tmp_path, model_name, model_link)

    # stdout, _ = capsys.readouterr()
    # assert "___TotalSequenceLengthPattern___" in stdout