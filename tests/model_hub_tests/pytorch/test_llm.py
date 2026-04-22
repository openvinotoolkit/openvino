# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
from contextlib import contextmanager

import numpy as np
import platform
import pytest
import torch

from models_hub_common.utils import retry
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
    orig_cuda_device_count = torch.cuda.device_count
    orig_post_init_model = None
    orig_awq_post_init = None
    orig_gemm_forward = None
    orig_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)

    # Import GPTQ-related modules BEFORE faking CUDA availability.
    # gptqmodel performs CUDA device init at import time; if it sees
    # cuda.is_available()=True while there is no real GPU, it crashes
    # trying to create CUDA streams.
    try:
        import gptqmodel  # noqa: F401
    except ImportError:
        pass
    try:
        from transformers.quantizers.quantizer_gptq import GptqHfQuantizer

        orig_post_init_model = GptqHfQuantizer._process_model_after_weight_loading

        def _process_model_after_weight_loading_cpu(self, model, **kwargs):
            if self.pre_quantized:
                class StoreAttr(object):
                    pass
                model.quantize_config = StoreAttr()
                oq = self.optimum_quantizer
                model.quantize_config.desc_act = oq.desc_act
                # gptqmodel's HFKernelLinear needs wf_unsqueeze_* buffers
                # (registered by PackableQuantLinear.post_init) but we must
                # NOT call optimize() which repacks qweight/qzeros into a
                # layout incompatible with OV's GPTQ decompression pattern.
                # Instead: register buffers only and force "train" mode so
                # the forward uses dequantize_weight (no compiled kernel).
                #
                # Also convert GPTQ v1→v2 qzeros format: auto_gptq stores
                # qzeros as (zero_point - 1); gptqmodel's dequantize_weight
                # expects actual zero_point values (v2). For 4-bit int32
                # packed format, add 0x11111111 (+1 per nibble).
                try:
                    from gptqmodel.nn_modules.qlinear import PackableQuantLinear
                    for _, submodule in model.named_modules():
                        if isinstance(submodule, PackableQuantLinear):
                            # v1→v2 qzeros conversion (4-bit int32 only)
                            if submodule.bits == 4 and submodule.qzeros.dtype == torch.int32:
                                submodule.qzeros.data += 0x11111111
                            PackableQuantLinear.post_init(submodule)
                            submodule.linear_mode = "train"
                except ImportError:
                    pass
                if oq.desc_act and not oq.disable_exllama and oq.max_input_length is not None:
                    try:
                        from gptqmodel import exllama_set_max_input_length
                        model = exllama_set_max_input_length(model, oq.max_input_length)
                    except ImportError:
                        pass
            return model

        GptqHfQuantizer._process_model_after_weight_loading = _process_model_after_weight_loading_cpu
    except ImportError:
        pass

    # Patch AWQ quantizer to skip gptqmodel's post_init/optimize, which
    # repacks qweight/qzeros into a layout that dequantize_gemm cannot
    # handle.  _replace_awq_with_linear() will dequantize from the
    # original packed format instead.
    try:
        from transformers.quantizers.quantizer_awq import AwqQuantizer
        orig_awq_post_init = AwqQuantizer._process_model_after_weight_loading
        AwqQuantizer._process_model_after_weight_loading = lambda self, model, **kwargs: model
    except ImportError:
        pass

    try:
        # Patch gptqmodel's AWQ linear to work on CPU without compiled
        # kernels.  _replace_awq_with_linear() replaces these modules
        # before conversion, so this patch only serves as a safety fallback.
        from gptqmodel.nn_modules.qlinear import AWQuantLinear
        from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm as _awq_dequant
        _orig_awq_forward = AWQuantLinear.forward

        def _awq_forward_cpu(self, x):
            orig_dtype = x.dtype
            out_shape = x.shape[:-1] + (self.out_features,)
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float16)
            weight = _awq_dequant(self.qweight, self.qzeros, self.scales,
                                  self.bits, self.group_size)
            out = torch.matmul(x_flat, weight)
            if self.bias is not None:
                out.add_(self.bias)
            return out.reshape(out_shape).to(orig_dtype)

        AWQuantLinear.forward = _awq_forward_cpu
        orig_gemm_forward = _orig_awq_forward
    except ImportError:
        pass

    if orig_gemm_forward is None:
        try:
            # Fallback: patch autoawq GEMM module for older setups
            from awq.modules.linear.gemm import WQLinearMMFunction
            from awq.utils.packing_utils import dequantize_gemm

            def new_forward(
                ctx, x, qweight, qzeros, scales,
                w_bit=4, group_size=128, bias=None, out_features=0,
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

    # Fake CUDA availability AFTER all imports are done, so gptqmodel's
    # module-level CUDA init has already completed safely on CPU.
    torch.cuda.is_available = lambda: True
    torch.cuda.is_bf16_supported = lambda: False
    torch.cuda.get_device_capability = lambda *args, **kwargs: (9, 1)
    torch.cuda.device_count = lambda: 1

    return (orig_cuda_is_available, orig_cuda_is_bf16_supported, orig_cuda_get_device_capability, orig_cuda_device_count), orig_post_init_model, orig_awq_post_init, orig_gemm_forward, orig_default_dtype


def unpatch_gptq(orig_cuda_check, orig_post_init_model, orig_awq_post_init, orig_gemm_forward, orig_default_dtype):
    torch.cuda.is_available, torch.cuda.is_bf16_supported, torch.cuda.get_device_capability, torch.cuda.device_count = orig_cuda_check
    torch.set_default_dtype(orig_default_dtype)
    try:
        from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
        GptqHfQuantizer._process_model_after_weight_loading = orig_post_init_model
    except ImportError:
        pass
    if orig_awq_post_init is not None:
        try:
            from transformers.quantizers.quantizer_awq import AwqQuantizer
            AwqQuantizer._process_model_after_weight_loading = orig_awq_post_init
        except ImportError:
            pass
    if orig_gemm_forward is not None:
        try:
            from gptqmodel.nn_modules.qlinear import AWQuantLinear
            AWQuantLinear.forward = orig_gemm_forward
        except ImportError:
            try:
                from awq.modules.linear.gemm import WQLinearMMFunction
                WQLinearMMFunction.forward = orig_gemm_forward
            except ImportError:
                pass


def _replace_awq_with_linear(model):
    """Replace AWQ quantized linear layers with plain nn.Linear.

    Eagerly dequantizes packed int weights into float32 so the conversion
    graph contains only standard matmul ops (no bitwise dequantization or
    compiled kernels).
    """
    try:
        from gptqmodel.nn_modules.qlinear import AWQuantLinear
        from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
    except ImportError:
        return

    for name, module in list(model.named_modules()):
        if not isinstance(module, AWQuantLinear):
            continue
        weight = dequantize_gemm(
            module.qweight, module.qzeros, module.scales,
            module.bits, module.group_size,
        ).t().contiguous().float()  # dequantize_gemm returns (in, out); Linear expects (out, in)
        linear = torch.nn.Linear(
            module.in_features, module.out_features,
            bias=module.bias is not None, dtype=weight.dtype,
        )
        linear.weight = torch.nn.Parameter(weight, requires_grad=False)
        if module.bias is not None:
            linear.bias = torch.nn.Parameter(module.bias.data.clone(), requires_grad=False)
        # Replace module in parent
        parts = name.rsplit(".", 1)
        parent = model.get_submodule(parts[0]) if len(parts) == 2 else model
        setattr(parent, parts[-1], linear)


@contextmanager
def _patch_for_jit_trace():
    """Patch transformers internals for JIT trace compatibility.

    1. Wrap sdpa_mask/eager_mask to coerce q_length from a 0-d tensor
       (produced by torch.jit.trace) to a plain int, preventing the BC
       check in sdpa_mask from crashing on q_length.shape[0].
    2. Patch DynamicLayer.lazy_initialization to create 4D empty tensors
       instead of 1D, avoiding aten::cat shape mismatches during trace.
    """
    saved_masks = {}
    orig_lazy_init = None
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask, eager_mask

        def _wrap_mask_fn(orig_fn):
            def _patched(*args, **kwargs):
                if "q_length" in kwargs and isinstance(kwargs["q_length"], torch.Tensor):
                    kwargs["q_length"] = kwargs["q_length"].item()
                return orig_fn(*args, **kwargs)
            return _patched

        saved_masks = {"sdpa": sdpa_mask, "eager": eager_mask}
        ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa", _wrap_mask_fn(sdpa_mask))
        ALL_MASK_ATTENTION_FUNCTIONS.register("eager", _wrap_mask_fn(eager_mask))
    except (ImportError, AttributeError):
        pass
    try:
        from transformers.cache_utils import DynamicLayer
        orig_lazy_init = DynamicLayer.lazy_initialization

        def _lazy_init_4d(self, key_states, value_states=None):
            self.dtype, self.device = key_states.dtype, key_states.device
            shape = list(key_states.shape)
            shape[-2] = 0
            self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.is_initialized = True

        DynamicLayer.lazy_initialization = _lazy_init_4d
    except (ImportError, AttributeError):
        pass
    try:
        yield
    finally:
        for name, func in saved_masks.items():
            ALL_MASK_ATTENTION_FUNCTIONS.register(name, func)
        if orig_lazy_init is not None:
            DynamicLayer.lazy_initialization = orig_lazy_init


class DynamicCacheModelWrapper(torch.nn.Module):
    """Wraps a causal LM to convert tuple past_key_values <-> DynamicCache.

    Newer transformers (>=4.48) use DynamicCache by default for many models.
    This wrapper ensures the model accepts tuple PKV (for OV tracing
    compatibility) and returns tuple PKV.
    """

    def __init__(self, model, example_keys=None):
        super().__init__()
        self._wrapped = model

        # Build an ordered list of parameter names matching the example keys,
        # preserving the original forward signature order.
        orig_params = list(inspect.signature(model.forward).parameters)
        if example_keys is not None:
            ordered_keys = [k for k in orig_params if k in example_keys]
        else:
            ordered_keys = orig_params

        # Create a forward function that converts tuple PKV <-> DynamicCache
        def _forward(*args, **kwargs):
            from transformers.cache_utils import DynamicCache

            # When called with positional args (during JIT tracing), convert
            # to kwargs using the ordered parameter names so the underlying
            # model receives correctly-named arguments regardless of
            # intermediate parameters (like cache_position) in its signature.
            if args and not kwargs:
                for name, arg in zip(ordered_keys, args):
                    kwargs[name] = arg
                args = ()

            # Convert tuple PKV to DynamicCache
            if "past_key_values" in kwargs and isinstance(kwargs["past_key_values"], (tuple, list)):
                kwargs["past_key_values"] = DynamicCache(ddp_cache_data=kwargs["past_key_values"])

            outputs = model(**kwargs)

            def _dc_to_tuples(dc):
                return tuple((k, v) for k, v, _ in dc)

            if isinstance(outputs, (tuple, list)):
                outputs = type(outputs)(
                    _dc_to_tuples(v) if isinstance(v, DynamicCache) else v
                    for v in outputs
                )
            else:
                # ModelOutput (dict-like): convert DynamicCache PKV
                if hasattr(outputs, 'past_key_values'):
                    pkv = outputs.past_key_values
                    if isinstance(pkv, DynamicCache):
                        outputs['past_key_values'] = _dc_to_tuples(pkv)
                # Convert ModelOutput to tuple for JIT trace compatibility
                outputs = tuple(v for v in outputs.values() if v is not None)

            return outputs

        # Set clean __signature__ so OV's process_dict_inputs matches inputs
        # without needing a wrapper (avoids Cache type annotation exec failure).
        sig_params = [
            inspect.Parameter(k, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)
            for k in ordered_keys
        ]
        _forward.__signature__ = inspect.Signature(parameters=sig_params)
        _forward.__wrapped__ = model.forward
        self.forward = _forward

    @property
    def config(self):
        return self._wrapped.config


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
        self.cuda_available, self.gptq_postinit, self.awq_postinit, self.orig_gemm_forward, self.orig_default_dtype = None, None, None, None, None
        self.export_mode = False

    @retry(3, exceptions=(OSError,), delay=1)
    def load_model(self, name, type):
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

        model_cached = snapshot_download(name)
        try:
            config = AutoConfig.from_pretrained(model_cached)
        except Exception:
            config = {}
        model_kwargs = {}
        is_quant = is_quantized_model(config)

        if is_quant:
            self.cuda_available, self.gptq_postinit, self.awq_postinit, self.orig_gemm_forward, self.orig_default_dtype = patch_gptq()
            model_kwargs["dtype"] = torch.float32
            self.ov_config = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}
        else:
            model_kwargs["dtype"] = "auto"

        t = AutoTokenizer.from_pretrained(model_cached)
        self.model = AutoModelForCausalLM.from_pretrained(model_cached, **model_kwargs)
        # Replace AWQ quantized linear modules with plain nn.Linear (eagerly
        # dequantized weights) before any forward pass, to prevent gptqmodel
        # from repacking weights and to keep bitwise ops out of the graph.
        _replace_awq_with_linear(self.model)
        if is_quant:
            model = self.model
        else:
            model = copy.deepcopy(self.model).float()

        # Build example inputs based on what the model's forward accepts
        fwd_params = set(inspect.signature(self.model.forward).parameters)
        example = t("Some input text to verify that model works.",
                    return_tensors='pt').__dict__['data']
        if "past_key_values" in fwd_params:
            pkv, am = self.get_pkv(model, t)
            example["past_key_values"] = pkv
            example["attention_mask"] = torch.cat(
                [example["attention_mask"], am], -1)
        if "position_ids" in fwd_params:
            ids = torch.cumsum(example["attention_mask"] != 0, dim=1) - 1
            example["position_ids"] = ids[:, -example["input_ids"].shape[1]:]
        self.example = example

        # Wrap models for DynamicCache <-> tuple conversion (transformers >=4.48)
        try:
            from transformers.cache_utils import DynamicCache  # noqa: F401
            needs_wrapper = True
        except ImportError:
            needs_wrapper = False
        if needs_wrapper:
            example_keys = set(example.keys())
            if is_quant:
                self.model = DynamicCacheModelWrapper(self.model, example_keys)
                model = self.model
            else:
                self.model = DynamicCacheModelWrapper(self.model, example_keys)
                model = DynamicCacheModelWrapper(model, example_keys)

        return model

    def get_inputs_info(self, model_obj):
        target = model_obj._wrapped if isinstance(model_obj, DynamicCacheModelWrapper) else model_obj
        return list(inspect.signature(getattr(target, "forward", target.__call__)).parameters)

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
        # Detect actual model dtype from parameters (config.torch_dtype can be None)
        model_dtype = next(
            (p.dtype for p in self.model.parameters()), torch.float32)
        if model_dtype in [torch.float16, torch.bfloat16]:
            patch(self.model)
            is_patched = True
        # initialize model after patching
        self.model(**self.example)
        with _patch_for_jit_trace(), torch.no_grad():
            ovm = super().convert_model_impl(self.model)
        if is_patched:
            unpatch(self.model, "_openvino_module_extension_patch_orig_forward")
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
            # Patch DynamicLayer.lazy_initialization to create 4D empty
            # tensors (same fix as _patch_for_jit_trace) so that aten::cat
            # along axis=-2 works correctly during export.
            with _patch_for_jit_trace():
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
                         torch.cuda.get_device_capability,
                         torch.cuda.device_count) = self.cuda_available
                    ovm = convert_model(exported, verbose=True)
        finally:
            if is_quant_patched:
                unpatch_quantized_for_export(self.model)
        return ovm

    def teardown_method(self):
        self.export_mode = False
        # restore after gptq patching
        if self.cuda_available is not None:
            unpatch_gptq(self.cuda_available, self.gptq_postinit, self.awq_postinit, self.orig_gemm_forward, self.orig_default_dtype)
            self.cuda_available, self.gptq_postinit, self.awq_postinit, self.orig_gemm_forward, self.orig_default_dtype = None, None, None, None, None
        super().teardown_method()

    @staticmethod
    def get_pkv(model, tokenizer):
        from transformers.cache_utils import DynamicCache

        for_pkv = tokenizer("To get past key values",
                            return_tensors='pt').__dict__['data']
        with torch.no_grad():
            pkv = model(**for_pkv)[1]
        # Convert DynamicCache to tuple-of-tuples format for OV tracing
        if isinstance(pkv, DynamicCache):
            pkv = tuple((k, v) for k, v, _ in pkv)
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
        ("gpt_neox", "EleutherAI/pythia-1.4b"),
        ("gpt_neox_japanese", "abeja/gpt-neox-japanese-2.7b"),
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
            ("opt_gptq", "katuni4ka/opt-125m-gptq"),
        ]

    @pytest.mark.parametrize("type,name", get_supported_export_precommit_models())
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_export_model_precommit(self, name, type, ie_device):
        self.export_mode = True
        self.run(model_name=name, model_link=type, ie_device=ie_device)
