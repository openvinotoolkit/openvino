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


def _build_rms_norm_decomposition_subgraph(input_shape, last_dim, model_buffers, opcode_index):
    """Build the rms_norm decomposition subgraph implementing
       out = (x / sqrt(mean(x^2, axis=-1, keepdims=True) + eps)) * gamma

    The subgraph has two Parameters (data, gamma) and one Result, matching the
    call-site signature of the parent STABLEHLO_COMPOSITE op so the fallback
    can splice it cleanly. Constants (axes, eps) live in model-level buffers
    appended to ``model_buffers``; ``opcode_index`` maps op names to indices in
    the model-level operatorCodes table.
    """
    # Constants live in model-level buffers — append and remember the indices.
    axes_buf = schema_fb.BufferT()
    axes_buf.data = list(np.array([len(input_shape) - 1], dtype=np.int32).tobytes())
    model_buffers.append(axes_buf)
    axes_buf_idx = len(model_buffers) - 1

    eps_buf = schema_fb.BufferT()
    eps_buf.data = list(np.array([0.0], dtype=np.float32).tobytes())  # rewritten below
    model_buffers.append(eps_buf)
    eps_buf_idx = len(model_buffers) - 1

    # Tensors (subgraph-local indices)
    def mk_tensor(name, shape, dtype, buf_idx):
        t = schema_fb.TensorT()
        t.shape = list(shape)
        t.type = dtype
        t.buffer = buf_idx
        t.name = name.encode()
        return t

    reduced_shape = list(input_shape)
    reduced_shape[-1] = 1

    t_data = mk_tensor("data", input_shape, schema_fb.TensorType.FLOAT32, 0)        # 0
    t_gamma = mk_tensor("gamma", [last_dim], schema_fb.TensorType.FLOAT32, 0)       # 1
    t_axes = mk_tensor("axes", [1], schema_fb.TensorType.INT32, axes_buf_idx)       # 2
    t_eps = mk_tensor("eps", [1], schema_fb.TensorType.FLOAT32, eps_buf_idx)        # 3
    t_sq = mk_tensor("sq", input_shape, schema_fb.TensorType.FLOAT32, 0)            # 4
    t_mean = mk_tensor("mean", reduced_shape, schema_fb.TensorType.FLOAT32, 0)      # 5
    t_mean_eps = mk_tensor("mean_eps", reduced_shape, schema_fb.TensorType.FLOAT32, 0)  # 6
    t_rms = mk_tensor("rms", reduced_shape, schema_fb.TensorType.FLOAT32, 0)        # 7
    t_norm = mk_tensor("norm", input_shape, schema_fb.TensorType.FLOAT32, 0)        # 8
    t_out = mk_tensor("decomp_out", input_shape, schema_fb.TensorType.FLOAT32, 0)   # 9

    def mk_op(opname, inputs, outputs, builtin_opts_type=0, builtin_opts=None):
        op = schema_fb.OperatorT()
        op.opcodeIndex = opcode_index[opname]
        op.inputs = inputs
        op.outputs = outputs
        op.builtinOptionsType = builtin_opts_type
        op.builtinOptions = builtin_opts
        return op

    # MEAN with keepdims=True
    reducer_opts = schema_fb.ReducerOptionsT()
    reducer_opts.keepDims = True

    add_opts = schema_fb.AddOptionsT()
    add_opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE
    mul_opts = schema_fb.MulOptionsT()
    mul_opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE
    div_opts = schema_fb.DivOptionsT()
    div_opts.fusedActivationFunction = schema_fb.ActivationFunctionType.NONE

    op_square = mk_op("SQUARE", [0], [4])
    op_mean = mk_op("MEAN", [4, 2], [5], schema_fb.BuiltinOptions.ReducerOptions, reducer_opts)
    op_add = mk_op("ADD", [5, 3], [6], schema_fb.BuiltinOptions.AddOptions, add_opts)
    op_sqrt = mk_op("SQRT", [6], [7])
    op_div = mk_op("DIV", [0, 7], [8], schema_fb.BuiltinOptions.DivOptions, div_opts)
    op_mul = mk_op("MUL", [8, 1], [9], schema_fb.BuiltinOptions.MulOptions, mul_opts)

    sg = schema_fb.SubGraphT()
    sg.name = b"odml.rms_norm_decomposition"
    sg.tensors = [t_data, t_gamma, t_axes, t_eps, t_sq, t_mean, t_mean_eps, t_rms, t_norm, t_out]
    sg.inputs = [0, 1]
    sg.outputs = [9]
    sg.operators = [op_square, op_mean, op_add, op_sqrt, op_div, op_mul]
    return sg, eps_buf_idx


def build_stablehlo_composite_rms_norm_model(input_shape, epsilon, composite_name="odml.rms_norm"):
    """Build a TFLite flatbuffer model with a STABLEHLO_COMPOSITE op.

    ``composite_name`` selects the dispatch path in the frontend:
      * "odml.rms_norm" → exercises the hand-written translator that emits
        ov::decomposition::rms_norm.
      * any other value (e.g. "odml.rms_norm_fallback") → exercises the
        decomposition-subgraph fallback, which inlines the subgraph built below.

    Both paths must produce numerically equivalent results.

    The model has one dynamic input (data) and a constant gamma (non-trivial
    values). The decomposition subgraph implements the math equivalent of
    odml.rms_norm so the frontend's fallback path produces correct results
    when no hand-written translator is registered for the composite name.
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

    # Operator codes: STABLEHLO_COMPOSITE for the parent op + the primitives
    # used by the decomposition subgraph.
    def mk_opcode(builtin):
        oc = schema_fb.OperatorCodeT()
        oc.builtinCode = builtin
        # builtinCode is widened to int32 in newer schemas; deprecated_builtin_code
        # must mirror it (or PLACEHOLDER_FOR_GREATER_OP_CODES for codes > 127).
        if builtin > 127:
            oc.deprecatedBuiltinCode = schema_fb.BuiltinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES
        else:
            oc.deprecatedBuiltinCode = builtin
        oc.version = 1
        return oc

    decomp_ops = ["SQUARE", "MEAN", "ADD", "SQRT", "DIV", "MUL"]
    model.operatorCodes = [mk_opcode(schema_fb.BuiltinOperator.STABLEHLO_COMPOSITE)]
    opcode_index = {}
    for name in decomp_ops:
        model.operatorCodes.append(mk_opcode(getattr(schema_fb.BuiltinOperator, name)))
        opcode_index[name] = len(model.operatorCodes) - 1

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
    composite_opts.name = composite_name
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

    # Decomposition subgraph implementing the same math as odml.rms_norm.
    # Used by the frontend's fallback path when no hand-written translator
    # is registered for this composite_name.
    decomp_subgraph, eps_buf_idx = _build_rms_norm_decomposition_subgraph(
        input_shape, last_dim, model.buffers, opcode_index
    )
    # Write the actual epsilon value into the buffer reserved for it.
    model.buffers[eps_buf_idx].data = list(np.array([epsilon], dtype=np.float32).tobytes())

    model.subgraphs = [main_subgraph, decomp_subgraph]
    return model


def rms_norm_reference(input_data, gamma, epsilon):
    """Compute RMS normalization reference: x / sqrt(mean(x^2) + eps) * gamma."""
    mean_sq = np.mean(np.square(input_data), axis=-1, keepdims=True)
    rms = np.sqrt(mean_sq + epsilon)
    return (input_data / rms) * gamma


class TestTFLiteStableHLOCompositeRmsNorm:
    """Layer test for STABLEHLO_COMPOSITE rms_norm translation.

    Covers both the hand-written 'odml.rms_norm' translator (which emits
    ov::decomposition::rms_norm and is later fused by ov::pass::RMSFusion at
    plugin compile time) and the generic decomposition-subgraph fallback
    (triggered for any other composite_name; the frontend inlines the
    decomposition subgraph carried in the TFLite model).
    """

    def _run(self, composite_name, params, ie_device, precision, temp_dir, model_filename):
        shape = params['shape']
        epsilon = params['epsilon']

        # Build and write the TFLite model
        model_obj = build_stablehlo_composite_rms_norm_model(shape, epsilon, composite_name)
        model_path = os.path.join(temp_dir, model_filename)
        fb_utils.write_model(model_obj, model_path)

        # Load directly via OV frontend
        core = ov.Core()
        ov_model = core.read_model(model_path)

        # Both dispatch paths must lower to the canonical RMS-norm shape:
        # ReduceMean(x^2) followed by Add(eps) and a Multiply by gamma.
        # The square-root step is not asserted because the fallback path's
        # TFLite SQRT translator emits Power(x, 0.5) instead of a Sqrt op.
        # Numerical equivalence with the NumPy reference (below) is the
        # real correctness signal.
        op_types = [op.get_type_name() for op in ov_model.get_ordered_ops()]
        for required in ("ReduceMean", "Add", "Multiply"):
            assert required in op_types, (
                f"Expected {required} op in translated model, got: {op_types}"
            )

        # For non-FP32 precisions, exercise the FP16-compression IR round-trip
        # the standard CommonLayerTest._test pipeline applies: save with
        # compress_to_fp16=True and reload from disk before compile_model.
        # Without this step the precision parameter only loosens the tolerance
        # below and never validates the FP16 conversion path.
        if precision != 'FP32':
            ir_path = os.path.join(temp_dir, model_filename + '.xml')
            ov.save_model(ov_model, ir_path, compress_to_fp16=True)
            ov_model = core.read_model(ir_path)

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
    @pytest.mark.precommit
    def test_stablehlo_composite_rms_norm(self, params, ie_device, precision, temp_dir):
        """Exercise the hand-written 'odml.rms_norm' translator path."""
        self._run(
            composite_name="odml.rms_norm",
            params=params,
            ie_device=ie_device,
            precision=precision,
            temp_dir=temp_dir,
            model_filename='stablehlo_composite_rms_norm.tflite',
        )

    @pytest.mark.parametrize("params", test_params)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_stablehlo_composite_rms_norm_fallback(self, params, ie_device, precision, temp_dir):
        """Exercise the decomposition-subgraph fallback by using a composite_name
        the frontend has no hand-written translator for. The frontend must
        inline the decomposition subgraph and the result must still match the
        NumPy reference.
        """
        self._run(
            composite_name="odml.rms_norm_fallback",
            params=params,
            ie_device=ie_device,
            precision=precision,
            temp_dir=temp_dir,
            model_filename='stablehlo_composite_rms_norm_fallback.tflite',
        )
