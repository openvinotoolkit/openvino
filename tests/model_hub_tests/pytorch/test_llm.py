# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from models_hub_common.utils import retry
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable as patch
from openvino.frontend.pytorch.patch_model import unpatch_model as unpatch
from torch_utils import TestTorchConvertModel


def is_gptq_model(config):
    config_dict = config.to_dict() if not isinstance(config, dict) else config
    quantization_config = config_dict.get("quantization_config", None)
    return quantization_config and quantization_config["quant_method"] == "gptq"


def patch_gptq():
    orig_cuda_is_available = torch.cuda.is_available
    orig_cuda_is_bf16_supported = torch.cuda.is_bf16_supported
    orig_cuda_get_device_capability = torch.cuda.get_device_capability
    orig_post_init_model = None
    torch.set_default_dtype(torch.float32)
    torch.cuda.is_available = lambda: True
    torch.cuda.is_bf16_supported = lambda: False
    torch.cuda.get_device_capability = lambda n: (9, 1)

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
    return (orig_cuda_is_available, orig_cuda_is_bf16_supported, orig_cuda_get_device_capability), orig_post_init_model


def unpatch_gptq(orig_cuda_check, orig_post_init_model):
    from optimum.gptq import GPTQQuantizer
    torch.cuda.is_available, torch.cuda.is_bf16_supported, torch.cuda.get_device_capability = orig_cuda_check
    GPTQQuantizer.post_init_model = orig_post_init_model


def to_numpy(t):
    if t.dtype in [torch.bfloat16, torch.float16]:
        return t.to(torch.float32).numpy(force=True)
    return t.numpy(force=True)


def flattenize_tuples(list_input):
    unpacked_pt_res = []
    for r in list_input:
        if isinstance(r, (tuple, list)):
            unpacked_pt_res.extend(flattenize_tuples(r))
        else:
            unpacked_pt_res.append(r)
    return unpacked_pt_res


def flattenize_outputs(outputs):
    if not isinstance(outputs, dict):
        outputs = flattenize_tuples(outputs)
        return [to_numpy(i) for i in outputs]
    else:
        return dict((k, to_numpy(v)) for k, v in outputs.items())


# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestLLMModel(TestTorchConvertModel):
    def setup_class(self):
        self.infer_timeout = 1800
        self.cuda_available, self.gptq_postinit = None, None

    @retry(3, exceptions=(OSError,), delay=1)
    def load_model(self, name, type):
        model = None
        example = None
        try:
            config = AutoConfig.from_pretrained(name, trust_remote_code=True)
        except Exception:
            config = {}
        model_kwargs = {"torchscript": True, "trust_remote_code": True}
        is_gptq = is_gptq_model(config)
        is_gpt2 = name == "openai-community/gpt2"

        if is_gptq:
            self.cuda_available, self.gptq_postinit = patch_gptq()
            model_kwargs["torch_dtype"] = torch.float32
            self.ov_config = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}
        elif is_gpt2:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = "auto"

        t = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
        if is_gptq:
            model = self.model
        else:
            assert self.model.config.torch_dtype in [
                torch.float16, torch.bfloat16] or is_gpt2
            model = copy.deepcopy(self.model).float()

        example = t("Some input text to verify that model works.",
                    return_tensors='pt').__dict__['data']
        atype = type.replace("_gptq", "")
        if atype not in ["gptj", "starcoder2", "mpt"]:
            pkv, am = self.get_pkv(model, t)
            example["past_key_values"] = pkv
            example["attention_mask"] = torch.cat(
                [example["attention_mask"], am], -1)
        if atype not in ["opt", "falcon", "mbart", "mpt"]:
            ids = torch.cumsum(example["attention_mask"] != 0, dim=1) - 1
            example["position_ids"] = ids[:, -
                                          example["input_ids"].shape[1]:]
        self.example = example
        return model

    def get_inputs_info(self, model_obj):
        return list(inspect.signature(getattr(model_obj, "forward", model_obj.__call__)).parameters)

    def prepare_inputs(self, inputs_info):
        inputs = getattr(self, "inputs", self.example)
        filtered_keys = [i for i in inputs_info if i in inputs]
        res = []
        for k in filtered_keys:
            v = inputs[k]
            if isinstance(v, tuple):
                v_flatten = flattenize_outputs(v)
                if k == "past_key_values":
                    v_flatten = [v.astype(np.float32) for v in v_flatten]
                res.extend(v_flatten)
            else:
                res.append(v.numpy())
        return res

    def infer_fw_model(self, model_obj, inputs):
        inputs = getattr(self, "inputs", self.example)
        fw_outputs = model_obj(**inputs)
        return flattenize_outputs(fw_outputs)

    def convert_model_impl(self, model_obj):
        is_patched = False
        if getattr(self.model.config, "torch_dtype", None) in [torch.float16, torch.bfloat16]:
            patch(self.model)
            is_patched = True
        # initialize model after patching
        self.model(**self.example)
        with torch.no_grad():
            ovm = super().convert_model_impl(self.model)
        if is_patched:
            unpatch(self.model, "_openvino_module_extension_patch_orig_forward")
        #    model_obj.float()
        return ovm

    def teardown_method(self):
        # restore after gptq patching
        if self.cuda_available is not None:
            unpatch_gptq(self.cuda_available, self.gptq_postinit)
            self.cuda_available, self.gptq_postinit = None, None
        super().teardown_method()

    @staticmethod
    def get_pkv(model, tokenizer):
        for_pkv = tokenizer("To get past key values",
                            return_tensors='pt').__dict__['data']
        with torch.no_grad():
            pkv = model(**for_pkv)[1]

        return pkv, for_pkv["attention_mask"]

    @pytest.mark.parametrize("type,name", [
        ("opt_gptq", "katuni4ka/opt-125m-gptq"),
        ("llama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        ("gpt2", "openai-community/gpt2")
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_convert_model_precommit(self, name, type, ie_device):
        self.run(model_name=name, model_link=type, ie_device=ie_device)

    @pytest.mark.parametrize("type,name", [
        ("gpt_neox", "databricks/dolly-v2-3b"),
        ("gpt_neox_japanese", "rinna/japanese-gpt-neox-3.6b"),
        ("opt", "facebook/opt-1.3b"),
        ("phi", "microsoft/phi-2"),
        ("phi3", "microsoft/Phi-3-mini-4k-instruct"),
        ("qwen2", "Qwen/Qwen2-0.5B-Instruct"),
        ("stablelm", "stabilityai/stablelm-3b-4e1t"),
        ("llama_gptq", "TheBloke/Llama-2-7B-Chat-GPTQ"),
        ("bloom_gptq", "sbolouki/bloom-1b7-gptq"),
        ("cohere_gptq", "shuyuej/aya-23-8B-GPTQ"),
        ("mbart_gptq", "Shivam098/opt-translation"),
    ])
    @pytest.mark.nightly
    def test_convert_model_nightly(self, name, type, ie_device):
        self.run(model_name=name, model_link=type, ie_device=ie_device)

    # too big for nightly
    @pytest.mark.parametrize("type,name", [
        ("aquila", "BAAI/AquilaChat2-7B"),
        ("baichuan", "baichuan-inc/Baichuan2-7B-Base"),
        pytest.param("chatglm", "THUDM/chatglm3-6b",
                     marks=pytest.mark.xfail(reason="Accuracy validation failed")),
        ("falcon", "tiiuae/falcon-7b-instruct"),
        ("fuyu", "ybelkada/fuyu-8b-sharded"),
        ("gemma", "beomi/gemma-ko-7b"),
        ("gemma2", "SteelStorage/Tess-v2.5-Gemma-2-27B-alpha-st"),
        ("gpt_neox", "togethercomputer/RedPajama-INCITE-7B-Instruct"),
        ("gpt_neox", "EleutherAI/gpt-neox-20b"),
        ("llama", "togethercomputer/LLaMA-2-7B-32K"),
        ("mistral", "HuggingFaceH4/zephyr-7b-beta"),
        ("mpt", "mosaicml/mpt-7b"),
        ("starcoder2", "cognitivecomputations/dolphincoder-starcoder2-7b"),
        ("persimmon", "adept/persimmon-8b-base"),
        pytest.param("mistral_gptq", "TheBloke/em_german_leo_mistral-GPTQ",
                     marks=pytest.mark.xfail(reason="GPTQ QUANT_TYPE=cuda is not supported")),
        pytest.param("llama3_gptq", "TechxGenus/Meta-Llama-3-8B-GPTQ",
                     marks=pytest.mark.xfail(reason="GPTQ QUANT_TYPE=cuda is not supported")),
    ])
    def test_convert_model_very_large(self, name, type, ie_device):
        self.run(model_name=name, model_link=type, ie_device=ie_device)
