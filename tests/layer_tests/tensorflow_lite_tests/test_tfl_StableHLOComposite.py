# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
from flatbuffers import flexbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.tools import flatbuffer_utils as fb_utils

import openvino as ov

test_params = [
    {'shape': [1, 3, 32], 'epsilon': 1e-6},
    {'shape': [2, 64], 'epsilon': 1e-5},
    {'shape': [1, 4, 8, 16], 'epsilon': 1e-6},
]


def _make_gamma(last_dim, seed=123):
    """Generate deterministic non-trivial gamma weights for testing."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 2.0, [last_dim]).astype(np.float32)


def build_stablehlo_composite_rms_norm_model(input_shape, epsilon):
    """Build a TFLite flatbuffer model with STABLEHLO_COMPOSITE odml.rms_norm op.

    The model has one dynamic input (data) and a constant gamma (non-trivial values).
    """
    last_dim = input_shape[-1]
    gamma_data = _make_gamma(last_dim)

    model = schema_fb.ModelT()
    model.version = 3
    model.description = "StableHLO Composite RMS Norm test model"

    # Buffers: [0]=null, [1]=input (empty), [2]=gamma weights, [3]=output (empty)
    buf_null = schema_fb.BufferT()
    buf_input = schema_fb.BufferT()
    buf_gamma = schema_fb.BufferT()
    buf_output = schema_fb.BufferT()

    buf_gamma.data = list(gamma_data.tobytes())
    model.buffers = [buf_null, buf_input, buf_gamma, buf_output]

    # Operator code for STABLEHLO_COMPOSITE (opcode 206)
    op_code = schema_fb.OperatorCodeT()
    op_code.builtinCode = schema_fb.BuiltinOperator.STABLEHLO_COMPOSITE
    op_code.deprecatedBuiltinCode = schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES
    op_code.version = 1
    model.operatorCodes = [op_code]

    # Tensor 0: input
    t_input = schema_fb.TensorT()
    t_input.shape = list(input_shape)
    t_input.type = schema_fb.TensorType.FLOAT32
    t_input.buffer = 1
    t_input.name = b"Input"

    # Tensor 1: gamma (constant, non-trivial values to verify correct wiring)
    t_gamma = schema_fb.TensorT()
    t_gamma.shape = [last_dim]
    t_gamma.type = schema_fb.TensorType.FLOAT32
    t_gamma.buffer = 2
    t_gamma.name = b"gamma"

    # Tensor 2: output
    t_output = schema_fb.TensorT()
    t_output.shape = list(input_shape)
    t_output.type = schema_fb.TensorType.FLOAT32
    t_output.buffer = 3
    t_output.name = b"RmsNormOutput"

    # STABLEHLO_COMPOSITE operator with odml.rms_norm options
    op = schema_fb.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0, 1]
    op.outputs = [2]
    op.builtinOptionsType = 0
    op.builtinOptions = None
    op.builtinOptions2Type = schema_fb.BuiltinOptions2.StableHLOCompositeOptions

    composite_opts = schema_fb.StableHLOCompositeOptionsT()
    composite_opts.name = "odml.rms_norm"
    composite_opts.decompositionSubgraphIndex = 1
    composite_opts.version = 1

    # Encode epsilon as FlexBuffer map
    fbb = flexbuffers.Builder()
    fbb.MapFromElements({"epsilon": float(epsilon)})
    composite_opts.compositeAttributes = list(fbb.Finish())
    composite_opts.compositeAttributesFormat = schema_fb.CustomOptionsFormat.FLEXBUFFERS
    op.builtinOptions2 = composite_opts

    # Main subgraph
    main_subgraph = schema_fb.SubGraphT()
    main_subgraph.tensors = [t_input, t_gamma, t_output]
    main_subgraph.inputs = [0]
    main_subgraph.outputs = [2]
    main_subgraph.operators = [op]
    main_subgraph.name = b"main"

    # Decomposition subgraph placeholder (required by schema)
    decomp_subgraph = schema_fb.SubGraphT()
    decomp_subgraph.name = b"odml.rms_norm_decomposition"
    t_decomp = schema_fb.TensorT()
    t_decomp.shape = list(input_shape)
    t_decomp.type = schema_fb.TensorType.FLOAT32
    t_decomp.buffer = 0
    t_decomp.name = b"decomp_placeholder"
    decomp_subgraph.tensors = [t_decomp]
    decomp_subgraph.inputs = [0]
    decomp_subgraph.outputs = [0]
    decomp_subgraph.operators = []

    model.subgraphs = [main_subgraph, decomp_subgraph]
    return model


def rms_norm_reference(input_data, gamma, epsilon):
    """Compute RMS normalization reference: x / sqrt(mean(x^2) + eps) * gamma."""
    mean_sq = np.mean(np.square(input_data), axis=-1, keepdims=True)
    rms = np.sqrt(mean_sq + epsilon)
    return (input_data / rms) * gamma


class TestTFLiteStableHLOCompositeRmsNorm:
    """Layer test for STABLEHLO_COMPOSITE odml.rms_norm op translation.

    This test builds a TFLite model containing a STABLEHLO_COMPOSITE operator
    with odml.rms_norm composite name, loads it directly via OV read_model
    (bypassing IR serialization since RMS is an internal op), runs inference,
    and compares against a NumPy reference implementation.
    """

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_stablehlo_composite_rms_norm(self, params, ie_device, precision, temp_dir):
        shape = params['shape']
        epsilon = params['epsilon']

        # Build and write the TFLite model
        model_obj = build_stablehlo_composite_rms_norm_model(shape, epsilon)
        model_path = os.path.join(temp_dir, 'stablehlo_composite_rms_norm.tflite')
        fb_utils.write_model(model_obj, model_path)

        # Load directly via OV frontend (no IR serialization needed for internal RMS op)
        core = ov.Core()
        ov_model = core.read_model(model_path)

        # Verify the model was translated to contain an RMS op
        op_types = [op.get_type_name() for op in ov_model.get_ordered_ops()]
        assert "RMS" in op_types, (
            f"Expected RMS op in translated model, got: {op_types}"
        )

        # Compile and infer
        config = {}
        if ie_device == 'GPU' and precision == 'FP32':
            config = {'INFERENCE_PRECISION_HINT': 'f32'}
        compiled_model = core.compile_model(ov_model, ie_device, config)

        # Generate random input
        rng = np.random.default_rng(42)
        input_data = rng.uniform(-1.0, 1.0, shape).astype(np.float32)
        gamma = _make_gamma(shape[-1])

        # Run OV inference
        result = compiled_model([input_data])
        ov_output = result[compiled_model.output(0)]

        # Compute reference
        expected = rms_norm_reference(input_data, gamma, epsilon)

        # Compare
        custom_eps = 1e-4 if precision == 'FP32' else 5e-2
        assert np.allclose(ov_output, expected, atol=custom_eps), (
            f"RMS norm output mismatch. Max diff: {np.max(np.abs(ov_output - expected)):.2e}, "
            f"tolerance: {custom_eps:.2e}"
        )

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    def test_stablehlo_composite_rms_norm_model_structure(self, params, ie_device, precision,
                                                          temp_dir):
        """Verify OV frontend correctly translates the STABLEHLO_COMPOSITE model structure."""
        shape = params['shape']
        epsilon = params['epsilon']

        model_obj = build_stablehlo_composite_rms_norm_model(shape, epsilon)
        model_path = os.path.join(temp_dir, 'stablehlo_composite_rms_norm.tflite')
        fb_utils.write_model(model_obj, model_path)

        core = ov.Core()
        ov_model = core.read_model(model_path)

        # Verify input/output shapes
        assert len(ov_model.inputs) == 1
        assert list(ov_model.inputs[0].shape) == shape

        assert len(ov_model.outputs) == 1
        assert list(ov_model.outputs[0].shape) == shape

        # Verify RMS op has correct epsilon
        for op in ov_model.get_ordered_ops():
            if op.get_type_name() == "RMS":
                attrs = op.get_attributes()
                assert abs(attrs['epsilon'] - epsilon) < 1e-10, (
                    f"Expected epsilon={epsilon}, got {attrs['epsilon']}"
                )
                break
        else:
            pytest.fail("RMS op not found in the model")
