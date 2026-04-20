# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect

import numpy as np
import platform
import pytest
import torch

from models_hub_common.utils import cached_snapshot_download, retry
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable as patch
from openvino.frontend.pytorch.patch_model import unpatch_model as unpatch
from torch_utils import TestTorchConvertModel


def is_quantized_model(config):
    config_dict = config.to_dict() if not isinstance(config, dict) else config
    quantization_config = config_dict.get("quantization_config", None)
    return quantization_config and quantization_config["quant_method"] in ["gptq", "awq"]


def patch_gptq():
    orig_cuda_is_available = torch.cuda.is_available
    orig_cuda_is_bf16_supported = torch.cuda.is_bf16_supported
    orig_cuda_get_device_capability = torch.cuda.get_device_capability
    orig_post_init_model = None
    orig_gemm_forward = None
    torch.set_default_dtype(torch.float32)
    torch.cuda.is_available = lambda: True
    torch.cuda.is_bf16_supported = lambda: False
    torch.cuda.get_device_capability = lambda n: (9, 1)

    try:
        # Patch at the transformers level to avoid GPU-only post_init_model
        # from optimum.gptq.  transformers' GptqHfQuantizer delegates to
        # optimum_quantizer.post_init_model() which calls gptq_post_init
        # (requires GPU).  Replace with a CPU-safe stub.
        from transformers.quantizers.quantizer_gptq import GptqHfQuantizer

        orig_post_init_model = GptqHfQuantizer._process_model_after_weight_loading

        def _process_model_after_weight_loading_cpu(self, model, **kwargs):
            if self.pre_quantized:
                class StoreAttr(object):
                    pass
                model.quantize_config = StoreAttr()
                oq = self.optimum_quantizer
                model.quantize_config.desc_act = oq.desc_act
                if oq.desc_act and not oq.disable_exllama and oq.max_input_length is not None:
                    try:
                        from auto_gptq import exllama_set_max_input_length
                        model = exllama_set_max_input_length(model, oq.max_input_length)
                    except ImportError:
                        pass
            return model

        GptqHfQuantizer._process_model_after_weight_loading = _process_model_after_weight_loading_cpu
    except ImportError:
        pass

    try:
        # patch GEMM module to work without CUDA GPU
        from awq.modules.linear.gemm import WQLinearMMFunction
        from awq.utils.packing_utils import dequantize_gemm

        def new_forward(
            ctx,
            x,
            qweight,
            qzeros,
            scales,
            w_bit=4,
            group_size=128,
            bias=None,
            out_features=0,
        ):
            ctx.out_features = out_features

            out_shape = x.shape[:-1] + (out_features,)
            x = x.to(torch.float16)

            out = dequantize_gemm(qweight, qzeros, scales, w_bit, group_size)
            out = torch.matmul(x, out.to(x.dtype))

            out = out + bias if bias is not None else out
            out = out.reshape(out_shape)

            if len(out.shape) == 2:
                out = out.unsqueeze(0)
            return out

        orig_gemm_forward = WQLinearMMFunction.forward
        WQLinearMMFunction.forward = new_forward
    except ImportError:
        pass
    return (orig_cuda_is_available, orig_cuda_is_bf16_supported, orig_cuda_get_device_capability), orig_post_init_model, orig_gemm_forward


def unpatch_gptq(orig_cuda_check, orig_post_init_model, orig_gemm_forward):
    torch.cuda.is_available, torch.cuda.is_bf16_supported, torch.cuda.get_device_capability = orig_cuda_check
    try:
        from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
        GptqHfQuantizer._process_model_after_weight_loading = orig_post_init_model
    except ImportError:
        pass
    try:
        from awq.modules.linear.gemm import WQLinearMMFunction
        WQLinearMMFunction.forward = orig_gemm_forward
    except ImportError:
        pass


def to_numpy(t):
    if t.dtype in [torch.bfloat16, torch.float16]:
        return t.to(torch.float32).numpy(force=True)
    return t.numpy(force=True)


def flattenize_tuples(list_input):
    unpacked_pt_res = []
    for r in list_input:
        if isinstance(r, (tuple, list)):
            unpacked_pt_res.extend(flattenize_tuples(r))
        elif hasattr(r, 'to_legacy_cache'):
            # DynamicCache → legacy tuple of (key, value) tuples
            unpacked_pt_res.extend(flattenize_tuples(r.to_legacy_cache()))
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
        self.cuda_available, self.gptq_postinit, self.orig_gemm_forward = None, None, None
        self.export_mode = False

    @retry(3, exceptions=(OSError,), delay=1)
    def load_model(self, name, type):
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

        model = None
        example = None
        model_cached = cached_snapshot_download(name)
        try:
            config = AutoConfig.from_pretrained(model_cached, trust_remote_code=True)
        except Exception:
            config = {}
        model_kwargs = {"torchscript": True, "trust_remote_code": True}
        is_quant = is_quantized_model(config)
        is_gpt2 = name == "openai-community/gpt2"

        if is_quant:
            self.cuda_available, self.gptq_postinit, self.orig_gemm_forward = patch_gptq()
            model_kwargs["torch_dtype"] = "auto"
            model_kwargs["torch_dtype"] = torch.float32
            self.ov_config = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}
        elif is_gpt2:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = "auto"
        model_cached = cached_snapshot_download(name)
        t = AutoTokenizer.from_pretrained(model_cached, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_cached, **model_kwargs)
        if is_quant:
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
            # DynamicCache (newer transformers) → legacy tuple for flattening
            if hasattr(v, 'to_legacy_cache'):
                v = v.to_legacy_cache()
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

    def infer_ov_model(self, ov_model, inputs, ie_device):
        if self.export_mode:
            return self._infer_ov_model_export(ov_model, ie_device)
        return super().infer_ov_model(ov_model, inputs, ie_device)

    def compare_results(self, fw_outputs, ov_outputs):
        if self.export_mode and isinstance(fw_outputs, (list, tuple)):
            # In export mode, only compare the first output (logits).
            # KV cache outputs diverge for quantized models because
            # FW and OV dequantization paths differ.
            return super().compare_results([fw_outputs[0]], [ov_outputs[0]])
        return super().compare_results(fw_outputs, ov_outputs)

    def _infer_ov_model_export(self, ov_model, ie_device):
        """Build inputs dict by OV model input names for the export path.

        FX export flattens nested structures (e.g. past_key_values tuples)
        into individual named inputs, so we match OV input names against
        the flattened example data.
        """
        from openvino import Core
        from torch.utils._pytree import tree_flatten

        example = getattr(self, "inputs", self.example)
        flat_values, _ = tree_flatten(example)
        flat_np = [to_numpy(v) if isinstance(v, torch.Tensor) else v
                   for v in flat_values]

        ov_inputs = {}
        for i, inp in enumerate(ov_model.inputs):
            if i < len(flat_np):
                ov_inputs[i] = flat_np[i]

        core = Core()
        compiled = core.compile_model(ov_model, ie_device, self.ov_config)
        return compiled(ov_inputs)

    def convert_model_impl(self, model_obj):
        if self.export_mode:
            return self._convert_model_export(model_obj)
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

    def _convert_model_export(self, model_obj):
        from torch.export import export
        from openvino import convert_model
        from openvino.frontend.pytorch.quantized import (
            detect_quantized_model,
            patch_quantized_for_export,
            unpatch_quantized_for_export,
        )

        is_quant_patched = False

        quant_type = detect_quantized_model(self.model)
        if quant_type:
            patch_quantized_for_export(self.model)
            is_quant_patched = True

        try:
            # Initialize model after patching
            self.model(**self.example)

            with torch.no_grad():
                exported = export(
                    self.model,
                    args=tuple(),
                    kwargs=self.example,
                    strict=False,
                )
                # Restore CUDA mocks before convert_model because
                # run_decompositions() tries to preserve CUDA RNG state
                # and fails when CUDA is mocked but not really available.
                # Only restore CUDA check functions — the AWQ GEMM forward
                # must stay patched for infer_fw_model.
                if self.cuda_available is not None:
                    (torch.cuda.is_available,
                     torch.cuda.is_bf16_supported,
                     torch.cuda.get_device_capability) = self.cuda_available
                ovm = convert_model(exported, verbose=True)
        finally:
            if is_quant_patched:
                unpatch_quantized_for_export(self.model)
        return ovm

    def teardown_method(self):
        self.export_mode = False
        # restore after gptq patching
        if self.cuda_available is not None:
            unpatch_gptq(self.cuda_available, self.gptq_postinit, self.orig_gemm_forward)
            self.cuda_available, self.gptq_postinit, self.orig_gemm_forward = None, None, None
        super().teardown_method()

    @staticmethod
    def get_pkv(model, tokenizer):
        for_pkv = tokenizer("To get past key values",
                            return_tensors='pt').__dict__['data']
        with torch.no_grad():
            pkv = model(**for_pkv)[1]

        # Newer transformers return DynamicCache; convert to legacy tuple format
        # so that the example dict contains plain tensors for both TS and FX paths.
        if hasattr(pkv, "to_legacy_cache"):
            pkv = pkv.to_legacy_cache()

        return pkv, for_pkv["attention_mask"]

    def get_supported_precommit_models():
        models = [
            ("gpt2", "openai-community/gpt2"),
        ]
        if platform.machine() not in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64']:
            models.extend([
                ("opt_gptq", "katuni4ka/opt-125m-gptq"),
                ("llama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                ("llama_awq", "casperhansen/tinyllama-1b-awq"),
            ])
        return models

    @pytest.mark.parametrize("type,name", get_supported_precommit_models())
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
        ("llama_awq", "TheBloke/open-llama-3b-v2-wizard-evol-instuct-v2-196k-AWQ")
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
        ("qwen2_awq", "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"),
        ("mixstral_awq", "TheBloke/SauerkrautLM-Mixtral-8x7B-AWQ"),
    ])
    def test_convert_model_very_large(self, name, type, ie_device):
        self.run(model_name=name, model_link=type, ie_device=ie_device)

    def get_supported_export_precommit_models():
        if platform.machine() in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64']:
            return []
        return [
            ("llama_awq", "casperhansen/tinyllama-1b-awq"),
        ]

    @pytest.mark.parametrize("type,name", get_supported_export_precommit_models())
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_export_model_precommit(self, name, type, ie_device):
        self.export_mode = True
        self.run(model_name=name, model_link=type, ie_device=ie_device)
