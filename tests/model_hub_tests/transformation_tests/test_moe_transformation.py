# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from huggingface_hub import snapshot_download
from optimum.intel import OVModelForCausalLM
import openvino as ov
from models_hub_common.utils import retry
import models_hub_common.utils as utils
import pytest
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

pytestmark = pytest.mark.skip(reason="FuseMOE transformation temporarily disabled in moc_transformations.cpp")


def verify_moe_fusion(ov_model: ov.Model, model_id: str):
    """
    Verify that MoE fusion was applied correctly by checking for fused weight tensors.
    
    After MoE fusion, we expect to find MatMul operations with weights that have
    an expert dimension (first dimension should equal the number of experts).
    
    Returns:
        int: Number of fused MoE layers detected
    """
    # Get model configuration to determine number of experts
    # For tiny-random-qwen3_moe, we expect num_experts parameter in config
    num_experts = None
    for op in ov_model.get_ordered_ops():
        # Look for patterns indicating fused MoE: 
        # - MatMul with 3D weight tensor [num_experts, hidden_dim, intermediate_dim]
        # - Tile operation that replicates input for all experts
        if op.get_type_name() == "MatMul":
            # Check if this MatMul has a 3D constant weight (indicating fused experts)
            for i in range(op.get_input_size()):
                input_node = op.input_value(i).get_node()
                if input_node.get_type_name() == "Constant":
                    weight_shape = list(input_node.get_shape())
                    # Fused expert weights should be 3D: [num_experts, in_dim, out_dim]
                    if len(weight_shape) == 3:
                        if num_experts is None:
                            num_experts = weight_shape[0]
                        else:
                            assert weight_shape[0] == num_experts, \
                                f"Inconsistent expert count: expected {num_experts}, got {weight_shape[0]}"
                # Check for Convert->Constant pattern (decompression)
                elif input_node.get_type_name() == "Convert":
                    convert_input = input_node.input_value(0).get_node()
                    if convert_input.get_type_name() == "Constant":
                        weight_shape = list(convert_input.get_shape())
                        if len(weight_shape) == 3:
                            if num_experts is None:
                                num_experts = weight_shape[0]
                            else:
                                assert weight_shape[0] == num_experts, \
                                    f"Inconsistent expert count: expected {num_experts}, got {weight_shape[0]}"
    
    # If we found fused weights, verify the number of experts makes sense
    if num_experts is not None:
        assert num_experts > 1, f"Expected multiple experts, found {num_experts}"
        # For tiny-random-qwen3_moe, we expect 4 experts
        print(f"Detected {num_experts} experts in fused MoE layers")
        return num_experts
    
    return 0


def verify_fused_moe_pattern(ov_model: ov.Model,
                            model_id: str,
                            ie_device: str):
    """
    Verify that the model has the fused MoE pattern.
    
    Since MOC transformations (including FuseMOE) are applied during model loading,
    we don't need to apply them again. Instead, we verify that the loaded model
    already contains the characteristic fused MoE subgraph with 3D weight tensors.
    
    This function:
    1. Verifies MoE fusion by checking for 3D fused weight tensors
    2. Compiles the model to ensure validity
    """
    # Verify that MoE fusion is present in the loaded model
    num_experts = verify_moe_fusion(ov_model, model_id)
    assert num_experts > 0, "Model does not contain fused MoE pattern with 3D weight tensors"
    
    # Compile to ensure the model is valid
    ov.Core().compile_model(ov_model, ie_device)


def create_synthetic_moe_model(tmp_path, num_layers, num_experts, dtype="float32", hidden_size=512, intermediate_size=192):
    """
    Create a synthetic Qwen3 MoE model from config for testing.
    
    Args:
        tmp_path: Temporary directory path
        num_layers: Number of hidden layers (MoE layers)
        num_experts: Number of experts per MoE layer
        dtype: Model dtype (float32, bfloat16, float16)
        hidden_size: Hidden dimension size
        intermediate_size: MoE intermediate size
    
    Returns:
        str: Path to the saved model
    """
    # Load config from cache to avoid HuggingFace rate limits
    config_cache = snapshot_download("optimum-internal-testing/tiny-random-qwen3_moe")
    config = AutoConfig.from_pretrained(config_cache)
    config.num_hidden_layers = num_layers
    config.decoder_sparse_step = 1
    config.num_experts = num_experts
    config.torch_dtype = dtype
    config.hidden_size = hidden_size
    config.moe_intermediate_size = intermediate_size
    
    model = AutoModelForCausalLM.from_config(config)
    model_path = os.path.join(tmp_path, f"synthetic_qwen3_moe_l{num_layers}_e{num_experts}_{dtype}")
    model.save_pretrained(model_path)
    
    return model_path


@retry(3, exceptions=(OSError,), delay=1)
def run_moe(tmp_path,
            model_id,
            model_link,
            ie_device):
    """
    Test that MoE models are loaded with fused expert subgraphs.
    
    MOC transformations (including FuseMOE) are automatically applied during
    model loading with from_pretrained, so we verify that the loaded model
    contains the characteristic fused MoE pattern.
    
    Additionally verifies output correctness by comparing with original PyTorch model.
    """
    model_cached = snapshot_download(model_id)  # required to avoid HF rate limits
    
    # Load original PyTorch model and tokenizer for comparison (from cache to avoid rate limits)
    pt_model = AutoModelForCausalLM.from_pretrained(model_cached, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_cached, trust_remote_code=True)
    
    # Prepare test input
    test_text = "Test input for MoE model verification"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    # Get PyTorch output
    with torch.no_grad():
        pt_outputs = pt_model(**inputs)
        pt_logits = pt_outputs.logits.numpy()
    
    # Clean up PyTorch model to free memory
    del pt_model
    del pt_outputs
    
    # Load the OpenVINO model - MOC transformations are applied during loading
    ov_model = OVModelForCausalLM.from_pretrained(
        model_cached, 
        export=True, 
        trust_remote_code=True,
        compile=False  # Don't compile yet, but transformations are still applied
    )
    
    # Verify that the loaded model has the fused MoE pattern
    verify_fused_moe_pattern(ov_model.model, model_id, ie_device)
    
    # Get OpenVINO output and compare with PyTorch
    ov_outputs = ov_model(**inputs)
    ov_logits = ov_outputs.logits.numpy()
    
    # Compare outputs (allow small numerical differences due to precision)
    max_diff = np.abs(pt_logits - ov_logits).max()
    mean_diff = np.abs(pt_logits - ov_logits).mean()
    
    print(f"Output comparison: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    # Verify outputs are close
    # Tolerances: rtol=1e-3, atol=1e-3 account for OpenVINO IR conversion and execution differences
    assert np.allclose(pt_logits, ov_logits, rtol=1e-3, atol=1e-3), \
        f"Output mismatch between PyTorch and OpenVINO fused model: max_diff={max_diff}, mean_diff={mean_diff}"


@retry(3, exceptions=(OSError,), delay=1)
def run_moe_synthetic(tmp_path,
                     num_layers,
                     num_experts,
                     dtype,
                     ie_device):
    """
    Test MoE fusion on synthetically generated models.
    
    Creates a model from config with specified parameters and verifies
    that MoE fusion produces the expected fused pattern.
    
    Additionally verifies output correctness by comparing with original PyTorch model.
    """
    model_path = create_synthetic_moe_model(tmp_path, num_layers, num_experts, dtype)
    
    # Load original PyTorch model for comparison (from local path)
    pt_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    # Load tokenizer from cache to avoid HuggingFace rate limits
    tokenizer_cache = snapshot_download("optimum-internal-testing/tiny-random-qwen3_moe")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cache, trust_remote_code=True)
    
    # Prepare test input
    test_text = "Test input for synthetic MoE model"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    # Get PyTorch output
    with torch.no_grad():
        pt_outputs = pt_model(**inputs)
        pt_logits = pt_outputs.logits.numpy()
    
    # Clean up PyTorch model to free memory
    del pt_model
    del pt_outputs
    
    # Load and export the OpenVINO model with MoE fusion
    ov_model = OVModelForCausalLM.from_pretrained(
        model_path,
        export=True,
        trust_remote_code=True,
        compile=False
    )
    
    # Verify that the loaded model has the fused MoE pattern
    verify_fused_moe_pattern(ov_model.model, f"synthetic_l{num_layers}_e{num_experts}_{dtype}", ie_device)
    
    # Verify the expected number of experts
    detected_experts = verify_moe_fusion(ov_model.model, f"synthetic_l{num_layers}_e{num_experts}_{dtype}")
    assert detected_experts == num_experts, \
        f"Expected {num_experts} experts, but detected {detected_experts}"
    
    # Get OpenVINO output and compare with PyTorch
    ov_outputs = ov_model(**inputs)
    ov_logits = ov_outputs.logits.numpy()
    
    # Compare outputs (allow small numerical differences)
    max_diff = np.abs(pt_logits - ov_logits).max()
    mean_diff = np.abs(pt_logits - ov_logits).mean()
    
    print(f"Synthetic ({num_layers}L, {num_experts}E, {dtype}): max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    # Adjust tolerances based on dtype
    # FP16/BF16 have lower precision, so allow larger numerical differences
    rtol = 1e-2 if dtype in ["float16", "bfloat16"] else 1e-3
    atol = 1e-2 if dtype in ["float16", "bfloat16"] else 1e-3

    assert np.allclose(pt_logits, ov_logits, rtol=rtol, atol=atol), \
        f"Output mismatch for {num_layers}L, {num_experts}E, {dtype}: max_diff={max_diff}, mean_diff={mean_diff}"


MOE_PRECOMMIT_TEST_CASES = [
    (OVModelForCausalLM, *model_info_tuple) 
    for model_info_tuple in utils.get_models_list(
        os.path.join(os.path.dirname(__file__), "models", "hf-tiny-random-moe-models-precommit")
    )
]


def moe_test_idfn(entry):
    """Generate test ID from model info."""
    retval = "moe-"
    if entry[0] is OVModelForCausalLM:
        retval += "text-"
    else:
        raise ValueError(f"Unknown model class {entry[0]}")
    retval += entry[1]
    return retval


@pytest.mark.precommit
@pytest.mark.parametrize("model_info_tuple", MOE_PRECOMMIT_TEST_CASES, ids=moe_test_idfn)
def test_moe_precommit(tmp_path, model_info_tuple, ie_device):
    """Test MoE fusion transformation on precommit models."""
    model_class, model_name, model_link, mark, reason = model_info_tuple
    assert mark is None or mark == 'skip' or mark == 'xfail', \
        "Incorrect test case: {}, {}".format(model_name, model_link)
    if mark == 'skip':
        pytest.skip(reason)
    elif mark == 'xfail':
        pytest.xfail(reason)
    
    run_moe(tmp_path, model_name, model_link, ie_device)


# Synthetic test cases with different configurations
MOE_SYNTHETIC_TEST_CASES = [
    # (num_layers, num_experts, dtype)
    # Test different numbers of MoE layers
    (1, 4, "float32"),   # Single MoE layer
    (2, 4, "float32"),   # Two MoE layers
    (3, 4, "float32"),   # Three MoE layers
    
    # Test different numbers of experts
    (1, 2, "float32"),   # Minimal: 2 experts
    (1, 8, "float32"),   # More experts: 8
    (1, 16, "float32"),  # Many experts: 16
    
    # Test different dtypes (important for decompression pattern testing)
    (1, 4, "float16"),   # FP16 - may have decompression
    (1, 4, "bfloat16"),  # BF16 - may have decompression
    
    # Combined variations
    (2, 8, "float16"),   # Multiple layers + more experts + FP16
    (3, 16, "bfloat16"), # Multiple layers + many experts + BF16
]


def synthetic_test_idfn(entry):
    """Generate test ID for synthetic test cases."""
    num_layers, num_experts, dtype = entry
    return f"synthetic-l{num_layers}-e{num_experts}-{dtype}"


@pytest.mark.precommit
@pytest.mark.parametrize("test_params", MOE_SYNTHETIC_TEST_CASES, ids=synthetic_test_idfn)
def test_moe_synthetic(tmp_path, test_params, ie_device):
    """
    Test MoE fusion with synthetically generated models.
    
    This tests various configurations:
    - Different numbers of MoE layers (1, 2, 3)
    - Different numbers of experts (2, 4, 8, 16)
    - Different dtypes (float32, float16, bfloat16) to validate decompression handling
    """
    num_layers, num_experts, dtype = test_params
    run_moe_synthetic(tmp_path, num_layers, num_experts, dtype, ie_device)
