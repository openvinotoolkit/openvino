# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Generate crafted .tflite model files with out-of-bounds quantized_dimension
# field values. These models test that the TFLite frontend properly validates
# the quantized_dimension field in QuantizationParameters before using it as
# a vector index, preventing CWE-787 (Out-of-bounds Write).
#
# The vulnerability is in get_quant_shape() at tflite_quantize_resolver.cpp:67
# where shape[quantization->get_axis()] = size; uses the quantized_dimension
# directly as an array index without bounds checking.
#
# Model structure:
#   input [1,3] float32 -> QUANTIZE -> output [1,3] int8
#   Output tensor has per-channel quantization (3 scale values) with a malicious
#   quantized_dimension value. When TFLQuantizeReplacer processes the TFLQuantize
#   node wrapping the output tensor, it calls get_quant_shape() which uses
#   quantized_dimension as a vector index.
#
# Models generated:
#   oob_quant_dim/axis_exceeds_rank.tflite  - quantized_dimension=100 on rank-2 tensor
#   oob_quant_dim/negative_axis.tflite      - quantized_dimension=-1 on rank-2 tensor

import os
import sys

import flatbuffers


def create_float_vector(builder, values):
    """Create a FlatBuffer vector of float32 values."""
    builder.StartVector(4, len(values), 4)
    for v in reversed(values):
        builder.PrependFloat32(v)
    return builder.EndVector()


def create_int64_vector(builder, values):
    """Create a FlatBuffer vector of int64 values."""
    builder.StartVector(8, len(values), 8)
    for v in reversed(values):
        builder.PrependInt64(v)
    return builder.EndVector()


def build_tflite_with_quantization(quantized_dimension=0):
    """
    Build a minimal .tflite FlatBuffer model with per-channel quantization
    on the output tensor of a QUANTIZE operator.

    Model structure:
      - 2 tensors: input [1,3] float32 (no quantization), output [1,3] int8
        (per-channel quantization with 3 scale values and a malicious axis)
      - 3 buffers (0=empty sentinel, 1=for input, 2=for output)
      - 1 operator code (QUANTIZE, builtin_code=114)
      - 1 operator: inputs=[0], outputs=[1]
      - SubGraph inputs=[0], outputs=[1]

    The QUANTIZE operator converts float32 input to int8 output. The output
    tensor has per-channel quantization, so TFLQuantizeReplacer will process
    it and call get_quant_shape() with the malicious quantized_dimension.

    FlatBuffer table field indices (from schema.fbs):
      QuantizationParameters: 0=min, 1=max, 2=scale, 3=zero_point,
                              4=details_type, 5=details, 6=quantized_dimension
      Tensor: 0=shape, 1=type, 2=buffer, 3=name, 4=quantization, 5=is_variable,
              6=sparsity, 7=shape_signature, 8=has_rank, 9=variant_tensors
      Operator: 0=opcode_index, 1=inputs, 2=outputs, ...
      OperatorCode: 0=deprecated_builtin_code, 1=custom_code, 2=version,
                    3=builtin_code
      SubGraph: 0=tensors, 1=inputs, 2=outputs, 3=operators, 4=name
      Model: 0=version, 1=operator_codes, 2=subgraphs, 3=description, 4=buffers

    Parameters:
      quantized_dimension: The axis value for per-channel quantization.
          Valid range for a rank-2 tensor is [0, 1].
          Values outside this range trigger the vulnerability.
    """
    builder = flatbuffers.Builder(2048)

    # -- Buffers (3 empty buffers) --
    buffer_offsets = []
    for _ in range(3):
        builder.StartObject(1)  # Buffer has 1 field: data
        buffer_offsets.append(builder.EndObject())

    builder.StartVector(4, len(buffer_offsets), 4)
    for off in reversed(buffer_offsets):
        builder.PrependUOffsetTRelative(off)
    buffers_vec = builder.EndVector()

    # -- Operator Code: QUANTIZE (builtin_code=114) --
    builder.StartObject(4)
    builder.PrependInt8Slot(0, 114, 0)   # deprecated_builtin_code = QUANTIZE
    builder.PrependInt32Slot(2, 1, 1)    # version = 1
    builder.PrependInt32Slot(3, 114, 0)  # builtin_code = QUANTIZE
    opcode_offset = builder.EndObject()

    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(opcode_offset)
    opcodes_vec = builder.EndVector()

    # -- QuantizationParameters for output tensor (per-channel, 3 scales) --
    # This is the malicious quantization with out-of-bounds quantized_dimension.
    # 3 scale values make it per-channel (size > 1), which triggers get_quant_shape().
    out_scale_vec = create_float_vector(builder, [0.1, 0.2, 0.3])
    out_zp_vec = create_int64_vector(builder, [0, 0, 0])
    builder.StartObject(7)  # QuantizationParameters has 7 fields (union = 2 slots)
    builder.PrependUOffsetTRelativeSlot(2, out_scale_vec, 0)   # scale (3 values)
    builder.PrependUOffsetTRelativeSlot(3, out_zp_vec, 0)      # zero_point (3 values)
    builder.PrependInt32Slot(6, quantized_dimension, 0)        # quantized_dimension (OOB!)
    out_quant = builder.EndObject()

    # -- Tensors --
    # Tensor 0: input [1,3] float32 (no quantization)
    name0 = builder.CreateString("input")
    builder.StartVector(4, 2, 4)
    builder.PrependInt32(3)
    builder.PrependInt32(1)
    shape0_vec = builder.EndVector()

    builder.StartObject(11)  # Tensor has up to 11 fields
    builder.PrependUOffsetTRelativeSlot(0, shape0_vec, 0)  # shape = [1, 3]
    builder.PrependInt8Slot(1, 0, 0)                        # type = FLOAT32 (0)
    builder.PrependUint32Slot(2, 1, 0)                      # buffer index = 1
    builder.PrependUOffsetTRelativeSlot(3, name0, 0)        # name
    tensor0 = builder.EndObject()

    # Tensor 1: output [1,3] int8 with per-channel quantization (malicious axis)
    name1 = builder.CreateString("output")
    builder.StartVector(4, 2, 4)
    builder.PrependInt32(3)
    builder.PrependInt32(1)
    shape1_vec = builder.EndVector()

    builder.StartObject(11)
    builder.PrependUOffsetTRelativeSlot(0, shape1_vec, 0)   # shape = [1, 3]
    builder.PrependInt8Slot(1, 9, 0)                         # type = INT8 (9)
    builder.PrependUint32Slot(2, 2, 0)                       # buffer index = 2
    builder.PrependUOffsetTRelativeSlot(3, name1, 0)         # name
    builder.PrependUOffsetTRelativeSlot(4, out_quant, 0)     # quantization (per-channel!)
    tensor1 = builder.EndObject()

    # -- Tensors vector --
    builder.StartVector(4, 2, 4)
    builder.PrependUOffsetTRelative(tensor1)
    builder.PrependUOffsetTRelative(tensor0)
    tensors_vec = builder.EndVector()

    # -- Operator: QUANTIZE, inputs=[0], outputs=[1] --
    builder.StartVector(4, 1, 4)
    builder.PrependInt32(0)
    op_inputs_vec = builder.EndVector()

    builder.StartVector(4, 1, 4)
    builder.PrependInt32(1)
    op_outputs_vec = builder.EndVector()

    builder.StartObject(11)  # Operator
    builder.PrependUint32Slot(0, 0, 0)                        # opcode_index = 0
    builder.PrependUOffsetTRelativeSlot(1, op_inputs_vec, 0)  # inputs = [0]
    builder.PrependUOffsetTRelativeSlot(2, op_outputs_vec, 0) # outputs = [1]
    op_offset = builder.EndObject()

    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(op_offset)
    operators_vec = builder.EndVector()

    # -- SubGraph --
    builder.StartVector(4, 1, 4)
    builder.PrependInt32(0)
    sg_inputs_vec = builder.EndVector()

    builder.StartVector(4, 1, 4)
    builder.PrependInt32(1)
    sg_outputs_vec = builder.EndVector()

    sg_name = builder.CreateString("main")

    builder.StartObject(7)  # SubGraph
    builder.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)    # tensors
    builder.PrependUOffsetTRelativeSlot(1, sg_inputs_vec, 0)  # inputs = [0]
    builder.PrependUOffsetTRelativeSlot(2, sg_outputs_vec, 0) # outputs = [1]
    builder.PrependUOffsetTRelativeSlot(3, operators_vec, 0)  # operators
    builder.PrependUOffsetTRelativeSlot(4, sg_name, 0)        # name
    sg_offset = builder.EndObject()

    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(sg_offset)
    subgraphs_vec = builder.EndVector()

    desc = builder.CreateString("oob_quant_dim_test_model")

    # -- Model --
    builder.StartObject(8)
    builder.PrependUint32Slot(0, 3, 0)                          # version = 3
    builder.PrependUOffsetTRelativeSlot(1, opcodes_vec, 0)      # operator_codes
    builder.PrependUOffsetTRelativeSlot(2, subgraphs_vec, 0)    # subgraphs
    builder.PrependUOffsetTRelativeSlot(3, desc, 0)             # description
    builder.PrependUOffsetTRelativeSlot(4, buffers_vec, 0)      # buffers
    model_offset = builder.EndObject()

    builder.Finish(model_offset, b"TFL3")
    return bytes(builder.Output())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_directory>")
        sys.exit(1)

    path_to_model_dir = os.path.join(sys.argv[1], "oob_quant_dim")
    os.makedirs(path_to_model_dir, exist_ok=True)

    # 1. quantized_dimension=100 on rank-2 tensor [1,3] -> OOB write in get_quant_shape()
    # get_quant_shape() at line 67: shape[100] = 3 with vector of size 2
    model = build_tflite_with_quantization(quantized_dimension=100)
    with open(os.path.join(path_to_model_dir, 'axis_exceeds_rank.tflite'), 'wb') as f:
        f.write(model)

    # 2. quantized_dimension=-1 on rank-2 tensor [1,3] -> negative axis
    # get_quantization() in utils.cpp should reject negative axis at parse time
    model = build_tflite_with_quantization(quantized_dimension=-1)
    with open(os.path.join(path_to_model_dir, 'negative_axis.tflite'), 'wb') as f:
        f.write(model)
