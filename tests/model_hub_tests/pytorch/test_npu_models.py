# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from models_hub_common.constants import hf_cache_dir, clean_hf_cache_dir
from models_hub_common.utils import cleanup_dir, retry
from openvino import PartialShape, convert_model

from torch_utils import TestTorchConvertModel, flattenize_tuples

# To make tests reproducible we seed the random generator
torch.manual_seed(0)

NPU_MODELS = [
    dict(id="llama", source="hf-causal-lm",
         repo="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16, ram_gb=5),
    dict(id="qwen2", source="hf-causal-lm",
         repo="Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16, ram_gb=3),
    dict(id="qwen3", source="hf-causal-lm",
         repo="Qwen/Qwen3-0.6B", dtype=torch.float16, ram_gb=4),
    # granite-4.0-h-1b and SmolLM3 need transformers 4.53+, which breaks decoder
    # tracing here - see envs/npu_models.txt.
    dict(id="phi3", source="hf-causal-lm",
         repo="microsoft/Phi-3-mini-4k-instruct", dtype=torch.float16, ram_gb=17),

    dict(id="bert", source="hf", auto_class="AutoModel",
         repo="bert-base-uncased", dtype=torch.float32, ram_gb=2,
         inputs={"input_ids": [1, 128], "attention_mask": [1, 128]}),

    dict(id="whisper", source="hf-whisper",
         repo="openai/whisper-base", dtype=torch.float32, ram_gb=2),

    dict(id="siglip", source="hf", auto_class="AutoModel",
         repo="google/siglip-base-patch16-224", dtype=torch.float32, ram_gb=2,
         inputs={"input_ids": [1, 64], "pixel_values": [1, 3, 224, 224]}),

    dict(id="detr", source="hf", auto_class="AutoModelForObjectDetection",
         repo="facebook/detr-resnet-50", dtype=torch.float32, ram_gb=1,
         inputs={"pixel_values": [1, 3, 640, 640]}),
    dict(id="rt-detr", source="hf", auto_class="AutoModelForObjectDetection",
         repo="PekingU/rtdetr_r50vd", dtype=torch.float32, ram_gb=1,
         inputs={"pixel_values": [1, 3, 640, 640]}),
    dict(id="deeplabv3", source="torchvision",
         repo="deeplabv3_mobilenet_v3_large", dtype=torch.float32, ram_gb=1),
    dict(id="sam2-hiera", source="timm",
         repo="sam2_hiera_base_plus.fb_r896", dtype=torch.float32, ram_gb=3),

    dict(id="depth-anything", source="hf", auto_class="AutoModelForDepthEstimation",
         repo="depth-anything/Depth-Anything-V2-Small-hf", dtype=torch.float32, ram_gb=1,
         inputs={"pixel_values": [1, 3, 518, 518]}),
    dict(id="swin2sr", source="hf", auto_class="Swin2SRForImageSuperResolution",
         repo="caidas/swin2SR-classical-sr-x2-64", dtype=torch.float32, ram_gb=1,
         inputs={"pixel_values": [1, 3, 64, 64]}),
    dict(id="edsr", source="super-image",
         repo="eugenesiow/edsr-base", dtype=torch.float32, ram_gb=1),

    dict(id="easyocr-detector", source="easyocr",
         repo="detector", dtype=torch.float32, ram_gb=1),
    dict(id="easyocr-recognizer", source="easyocr",
         repo="recognizer", dtype=torch.float32, ram_gb=1),

    dict(id="resnet", source="torchvision",
         repo="resnet50", dtype=torch.float32, ram_gb=1),
    dict(id="convnext", source="torchvision",
         repo="convnext_tiny", dtype=torch.float32, ram_gb=1),
    dict(id="vgg19", source="torchvision",
         repo="vgg19", dtype=torch.float32, ram_gb=2),
    # The quantized variant: it drags in quantize/dequantize and quantized conv,
    # which no other entry here does.
    dict(id="mobilenet-v3", source="torchvision",
         repo="quantized_mobilenet_v3_large", dtype=torch.float32, ram_gb=1),
    dict(id="inception", source="timm",
         repo="inception_v4.tf_in1k", dtype=torch.float32, ram_gb=1),
    dict(id="mobilevit", source="timm",
         repo="mobilevit_s.cvnets_in1k", dtype=torch.float32, ram_gb=1),
    dict(id="hrnet", source="timm",
         repo="hrnet_w18.ms_aug_in1k", dtype=torch.float32, ram_gb=1),
]

pytestmark = pytest.mark.skipif("NPU" not in os.environ.get("TEST_DEVICE", ""),
                                reason="the list is NPU only")


def get_case(model_id: str) -> dict:
    # by id, not self: run() re-imports this module in a fresh process
    for case in NPU_MODELS:
        if case["id"] == model_id:
            return case
    raise RuntimeError(f"unknown model id: {model_id}")


def skip_if_not_enough_ram(model_name, need_gb):
    import psutil

    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    if available_gb < need_gb:
        pytest.skip(f"{model_name}: needs ~{need_gb} GB RAM to convert, "
                    f"only {available_gb:.1f} GB available")


def make_inputs(spec, dtype):
    inputs = {}
    for name, shape in spec.items():
        if name.endswith("_ids"):
            inputs[name] = torch.randint(0, 1000, shape)
        elif name.endswith("_mask"):
            inputs[name] = torch.ones(shape, dtype=torch.int64)
        else:
            inputs[name] = torch.randn(shape, dtype=dtype)
    return inputs


def flatten_tensors(value):
    # ModelOutput is dict-like; None and cache objects are dropped along the way.
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_tensors(item)


class _Traceable(torch.nn.Module):
    # torch.jit.trace wants positional tensors in and a flat tuple of tensors out;
    # HF outputs carry None fields and nested ModelOutput/cache objects it can't type.
    def __init__(self, module, names=None):
        super().__init__()
        self.module = module
        self.names = names

    def forward(self, *args):
        if self.names:
            out = self.module(**dict(zip(self.names, args)))
        else:
            out = self.module(*args)
        return tuple(flatten_tensors(out))


class TestNpuModels(TestTorchConvertModel):
    # weights download plus a compile of a multi-billion parameter model
    infer_timeout = 3600

    @retry(10, exceptions=(OSError,), delay=5, exponential_backoff=True,
           backoff_multiplier=2, max_delay=300)
    def load_model(self, model_name, model_link):
        case = get_case(model_name)
        loader = {
            "hf": self._load_hf,
            "hf-causal-lm": self._load_hf_causal_lm,
            "hf-whisper": self._load_hf_whisper,
            "timm": self._load_timm,
            "torchvision": self._load_torchvision,
            "super-image": self._load_super_image,
            "easyocr": self._load_easyocr,
        }[case["source"]]

        model = loader(case)
        model.eval()
        if isinstance(self.example, dict):
            names = list(self.example)
            self.example = tuple(self.example.values())
            return _Traceable(model, names)
        return _Traceable(model)

    def _load_hf(self, case):
        import transformers
        from huggingface_hub import snapshot_download

        cached = snapshot_download(case["repo"])  # required to avoid HF rate limits
        config = transformers.AutoConfig.from_pretrained(cached)
        config.torchscript = True
        # not every config has use_cache
        if hasattr(config, "use_cache"):
            config.use_cache = False
        model = getattr(transformers, case["auto_class"]).from_pretrained(
            cached, config=config, torch_dtype=case["dtype"])
        self.example = make_inputs(case["inputs"], case["dtype"])
        return model

    def _load_hf_causal_lm(self, case):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM

        cached = snapshot_download(case["repo"])
        # eager + use_cache=False: sdpa branching and DynamicCache do not trace
        model = AutoModelForCausalLM.from_pretrained(cached,
                                                     torch_dtype=case["dtype"],
                                                     torchscript=True,
                                                     attn_implementation="eager",
                                                     use_cache=False)
        self.example = make_inputs({"input_ids": [1, 32], "attention_mask": [1, 32]},
                                   case["dtype"])
        return model

    def _load_hf_whisper(self, case):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSpeechSeq2Seq

        cached = snapshot_download(case["repo"])
        model = AutoModelForSpeechSeq2Seq.from_pretrained(cached,
                                                          torch_dtype=case["dtype"],
                                                          torchscript=True,
                                                          attn_implementation="eager",
                                                          use_cache=False)
        # 3000 frames is whisper's fixed 30-second window; mel bins differ per variant
        self.example = {
            "input_features": torch.randn([1, model.config.num_mel_bins, 3000],
                                          dtype=case["dtype"]),
            "decoder_input_ids": torch.randint(0, 1000, [1, 20]),
        }
        return model

    def _load_timm(self, case):
        import timm

        model = timm.create_model(case["repo"], pretrained=True)
        shape = list(timm.get_pretrained_cfg(case["repo"]).input_size)
        self.example = (torch.randn([1] + shape, dtype=case["dtype"]),)
        return model

    def _load_torchvision(self, case):
        from torchvision.models import get_model

        model = get_model(case["repo"], weights="DEFAULT")
        self.example = (torch.randn([1, 3, 224, 224], dtype=case["dtype"]),)
        return model

    def _load_super_image(self, case):
        from super_image import EdsrModel

        model = EdsrModel.from_pretrained(case["repo"], scale=2)
        self.example = (torch.randn([1, 3, 64, 64], dtype=case["dtype"]),)
        return model

    def _load_easyocr(self, case):
        import easyocr

        reader = easyocr.Reader(["en"], quantize=False)
        if case["repo"] == "detector":
            model = reader.detector
            self.example = (torch.rand([1, 3, 608, 800], dtype=case["dtype"]),)
        else:
            model = reader.recognizer
            self.example = (torch.rand([1, 1, 64, 320], dtype=case["dtype"]),
                            torch.rand([1, 33], dtype=case["dtype"]))
        return model

    def convert_model_impl(self, model_obj):
        # NPU requires static shapes: use example input dims
        example = self.example
        tensors = list(example.values()) if isinstance(example, dict) else flattenize_tuples(example)
        return convert_model(model_obj,
                             example_input=example,
                             input=[PartialShape(list(t.shape)) for t in tensors],
                             verbose=True)

    def teardown_method(self):
        if clean_hf_cache_dir:
            cleanup_dir(hf_cache_dir)

        super().teardown_method()

    @pytest.mark.parametrize("model_id", [case["id"] for case in NPU_MODELS])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_compile_model(self, model_id, ie_device):
        case = get_case(model_id)
        skip_if_not_enough_ram(model_id, case["ram_gb"])
        self.run(model_id, case["repo"], ie_device)
