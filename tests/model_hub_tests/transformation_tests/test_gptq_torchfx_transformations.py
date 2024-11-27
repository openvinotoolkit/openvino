# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import hashlib
from openvino.frontend.pytorch.torchdynamo.execute import compiled_cache
import models_hub_common.utils as utils
import pytest
import os

def patch_gptq(config):
    do_gptq_patching = False
    config_dict = config.to_dict()
    quantization_config = config_dict.get("quantization_config", None)
    do_gptq_patching = quantization_config and quantization_config["quant_method"] == "gptq"
    orig_cuda_check = torch.cuda.is_available
    orig_post_init_model = None
    if do_gptq_patching:
        torch.set_default_dtype(torch.float32)
        torch.cuda.is_available = lambda: False

        from optimum.gptq import GPTQQuantizer

        orig_post_init_model = GPTQQuantizer.post_init_model

        def post_init_model(self, model):
            from auto_gptq import exllama_set_max_input_length

            class StoreAttr(object):
                pass

            model.quantize_config = StoreAttr()
            model.quantize_config.desc_act = self.desc_act
            if self.desc_act and not self.disable_exllama and self.max_input_length is not None:
                model = exllama_set_max_input_length(model, self.max_input_length)
            return model

        GPTQQuantizer.post_init_model = post_init_model
    return orig_cuda_check, orig_post_init_model

def run_gptq_torchfx(tmp_path, model_id, model_link, prompt_result_pair):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
    cuda, post_init = patch_gptq(config)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=config,
        device_map='cpu',
        torch_dtype=torch.float32
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4,
        do_sample=True,
        temperature=0.01,
        top_p=0.01,
        top_k=1,
        repetition_penalty=1.1,
        num_beams=1,
    )

    prompt = prompt_result_pair["prompt"]
    expected_md5 = prompt_result_pair["result_md5"]
    
    model.model.forward = torch.compile(model.model.forward, backend="openvino", dynamic=True, fullgraph=True, options={'aot_autograd': True})
    
    result_ov = pipe(prompt)
    md5_ov = hashlib.new("md5", result_ov[0]['generated_text'].encode(), usedforsecurity=False).hexdigest()
    
    u4_ops = ["FullyConnected",]
    num_u4_ops = 0
    num_u4_ops_supported = 0
    for pid in compiled_cache:
        for op in compiled_cache[pid].get_runtime_model().get_ordered_ops():
            if (str(op.get_rt_info()["layerType"].get()) in u4_ops):
                u4_exec = (str(op.get_rt_info()["runtimePrecision"].get()) == "u4")
                if u4_exec:
                    num_u4_ops_supported += 1
                num_u4_ops += 1
    
    assert(expected_md5 == md5_ov), "Output does not match with the expected output"
    assert((num_u4_ops > 0) and (num_u4_ops == num_u4_ops_supported)), "Runtime precision is not u4"

@pytest.mark.precommit
@pytest.mark.parametrize("model_name, model_link, mark, reason", utils.get_models_list(os.path.join(os.path.dirname(__file__), "models", "gptq-torchfx-models-precommit")))
@pytest.mark.parametrize('prompt_result_pair', ([
    {"prompt" : "Tell me about AI", "result_md5" : "4385ccbce14627ae91f846b4c8a3f145"},
]))
def test_gptq_torchfx_precommit(tmp_path, model_name, model_link, mark, reason, prompt_result_pair, ie_device):
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    run_gptq_torchfx(tmp_path, model_name, model_link, prompt_result_pair)

