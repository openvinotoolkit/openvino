# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from huggingface_hub import snapshot_download
from openvino._offline_transformations import paged_attention_transformation
from openvino._pyopenvino.op import _PagedAttentionExtension
from optimum.intel.openvino import OVModelForVisualCausalLM
import openvino as ov
from models_hub_common.utils import retry
import pytest
import platform

GEMMA4_MODEL_ID = "optimum-intel-internal-testing/tiny-random-gemma4"


@retry(3, exceptions=(OSError,), delay=1)
def get_gemma4_language_model(model_id):
    model_cached = snapshot_download(model_id)  # required to avoid HF rate limits
    # The optimum-intel version from envs/paged_attention_v5.txt exposes the language model via .language_model.model.
    model = OVModelForVisualCausalLM.from_pretrained(model_cached, export=True, trust_remote_code=True)
    return model.language_model.model


@pytest.mark.precommit
def test_pa_gemma4(ie_device):
    # gemma4's rotary embedding derives its batch dimension from attention_mask via a
    # Broadcast+MatMul; BroadcastMatMulFusion must remove that broadcast so
    # SDPAToPagedAttention can safely drop the attention_mask parameter.
    if platform.machine() in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64']:
        pytest.skip("PagedAttention tests are not enabled on ARM")

    ov_model = get_gemma4_language_model(GEMMA4_MODEL_ID)

    paged_attention_transformation(ov_model, False, False, False, False, False, False, False)
    ov.Core().compile_model(ov_model, ie_device)

    pa_count = sum(1 for op in ov_model.get_ordered_ops() if isinstance(op, _PagedAttentionExtension))
    assert pa_count > 0, "PagedAttentionExtension nodes were not created"
