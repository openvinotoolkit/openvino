# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# WHAT:
#   Generates a family of small .tflite test models in which a constant
#   tensor carries a non-null SparsityParameters table with sub-fields that
#   are either incomplete (missing/empty traversal_order, dim_metadata,
#   dim_format, or fully empty SparsityParameters) or that exercise the
#   "valid standard CSR" corner case where block_map is absent. The unifying
#   contract: every model produced here must both load() and convert()
#   without throwing once the OpenVINO TFLite frontend's get_sparsity()
#   factory invokes SparsityInfo::enable(). Two outcomes are expected:
#     - Incomplete metadata (missing/empty traversal_order, dim_metadata,
#       dim_format, or all-empty SparsityParameters) → tensor is recognized
#       as "not really sparse" and falls back to the raw constant buffer.
#     - Missing block_map on a non-block-sparse layout (traversal_order
#       length == tensor rank) → tensor is treated as a valid standard CSR
#       sparse tensor and is densified normally. block_map is required only
#       for block-sparse layouts (schema.fbs:187-211).
#
# HOW:
#   Uses the official Python `flatbuffers.Builder` (already a transitive
#   dependency of the TensorFlow package required by tests/requirements.txt,
#   so no new dep is introduced). Field offsets and table layouts follow
#   src/frontends/tensorflow_lite/src/schema/schema.fbs verbatim.
#   build_incomplete_sparsity_model() takes knobs for which of the three
#   SparsityParameters sub-fields to include (or to include but leave empty)
#   and which simple op graph to emit:
#       - "add"       : input + const_with_sparsity  -> output  (all FLOAT32)
#       - "dequantize": DEQUANTIZE(const_with_sparsity) -> mid;
#                       ADD(input, mid) -> output  (mirrors bug location)
#   For both graphs the constant tensor's buffer holds exactly
#   prod(shape) * elem_size bytes so that the buffer-size check in
#   load_model() passes once enable() has correctly disabled the sparsity
#   and TensorLitePlace consequently exposes the raw buffer.
#
# WHY:
#   Regression coverage for src/frontends/tensorflow_lite/src/utils.cpp's
#   get_sparsity() factory. Without sparsity->enable() at the end of that
#   factory, an incomplete SparsityParameters keeps m_disabled = false,
#   TensorLitePlace::ctor calls dense_data() -> densify() and the convert
#   step throws "Sparse dimension isn't found for sparse tensor". With the
#   fix, every model produced here loads + converts cleanly — either by
#   falling back to the raw buffer (incomplete-metadata cases) or by
#   densifying a valid standard CSR tensor (missing-block_map case).
#
#   The missing_block_map model deliberately uses segments=[0, 0, 0] and
#   indices=[] so that densify() produces an all-zero (2, 2) constant.
#   Its raw FLOAT32 buffer is [1, 2, 3, 4]. The two outcomes are therefore
#   observable on the converted model: ADD(input, const) == input proves
#   densify() ran (asserted by IncompleteSparsityDensify in
#   convert_sparse_incomplete.cpp), and a future regression that disabled
#   this tensor would visibly produce input + [1, 2, 3, 4] instead.
#
#   Real-world reproducer hand_landmark_full.tflite is 5.3 MB and cannot be
#   committed, so the synthetic flatbuffers below stand in for it.
#
# Models produced (in subdirectory sparse_incomplete/):
#   1. missing_dim_metadata.tflite           - traversal_order + block_map set; dim_metadata field omitted (HandsLandmarkFull shape)
#   2. empty_dim_metadata.tflite             - traversal_order + block_map set; dim_metadata is an empty vector
#   3. missing_traversal_order.tflite        - block_map + dim_metadata set; traversal_order omitted
#   4. missing_block_map.tflite              - traversal_order + dim_metadata set; block_map omitted
#   5. empty_sparsity.tflite                 - SparsityParameters table present but all three fields omitted
#   6. dequantize_incomplete_sparsity.tflite - INT8 const with empty SparsityParameters consumed by DEQUANTIZE then ADD
#
# Generated tflite files are NOT committed to git; they are produced by the
# CMake target 'tensorflow_lite_test_models' at build time.

import os
import struct
import sys

import flatbuffers

# tflite/schema.fbs enum values (mirrored from
# src/frontends/tensorflow_lite/src/schema/schema.fbs).
TT_FLOAT32 = 0
TT_INT8 = 9

OP_ADD = 0
OP_DEQUANTIZE = 6

DIM_DENSE = 0
DIM_SPARSE_CSR = 1

# SparseIndexVector union tags
SPARSE_IDX_NONE = 0
SPARSE_IDX_INT32 = 1  # Int32Vector

# BuiltinOptions union tags (from schema.fbs:union BuiltinOptions order)
BO_NONE = 0
BO_AddOptions = 11

# ActivationFunctionType
ACT_NONE = 0


# ---------- Helpers building schema.fbs tables ----------

def _create_int_vector(b, values, prepend_fn=None):
    """Create a Vector<int32> in the buffer, return offset."""
    prepend = prepend_fn or b.PrependInt32
    b.StartVector(4, len(values), 4)
    for v in reversed(values):
        prepend(v)
    return b.EndVector()


def _create_uint_vector(b, values):
    b.StartVector(4, len(values), 4)
    for v in reversed(values):
        b.PrependUint32(v)
    return b.EndVector()


def _create_byte_vector(b, payload):
    b.StartVector(1, len(payload), 1)
    for byte in reversed(payload):
        b.PrependUint8(byte)
    return b.EndVector()


def _create_offset_vector(b, offsets):
    b.StartVector(4, len(offsets), 4)
    for off in reversed(offsets):
        b.PrependUOffsetTRelative(off)
    return b.EndVector()


def _build_int32_vector_table(b, values):
    """schema.fbs:Int32Vector { values:[int]; } - field 0 only."""
    values_off = _create_int_vector(b, values) if values else _create_int_vector(b, [])
    b.StartObject(1)
    b.PrependUOffsetTRelativeSlot(0, values_off, 0)
    return b.EndObject()


def _build_dim_metadata_dense(b, dense_size):
    """schema.fbs:DimensionMetadata { format, dense_size, ... }."""
    b.StartObject(6)
    b.PrependInt8Slot(0, DIM_DENSE, 0)
    b.PrependInt32Slot(1, dense_size, 0)
    return b.EndObject()


def _build_dim_metadata_sparse_csr(b, segments_off, indices_off):
    """SPARSE_CSR DimensionMetadata referring to two Int32Vector tables."""
    b.StartObject(6)
    b.PrependInt8Slot(0, DIM_SPARSE_CSR, 0)
    b.PrependInt32Slot(1, 0, 0)  # dense_size unused for sparse dims
    b.PrependUint8Slot(2, SPARSE_IDX_INT32, 0)  # array_segments_type
    b.PrependUOffsetTRelativeSlot(3, segments_off, 0)  # array_segments
    b.PrependUint8Slot(4, SPARSE_IDX_INT32, 0)  # array_indices_type
    b.PrependUOffsetTRelativeSlot(5, indices_off, 0)  # array_indices
    return b.EndObject()


def _build_sparsity_params(b, *, include_traversal_order, include_block_map,
                           dim_metadata_mode):
    """
    SparsityParameters {
      traversal_order:[int];       // field 0
      block_map:[int];             // field 1
      dim_metadata:[DimensionMetadata]; // field 2
    }

    dim_metadata_mode:
        "full"    - 2 valid DimensionMetadata entries
        "empty"   - empty vector
        "missing" - field absent
    """
    if include_traversal_order:
        traversal_order_off = _create_int_vector(b, [0, 1])
    else:
        traversal_order_off = 0

    if include_block_map:
        block_map_off = _create_int_vector(b, [0])
    else:
        block_map_off = 0

    if dim_metadata_mode == "full":
        # Build segments / indices Int32Vector tables BEFORE starting any
        # other table (FlatBuffers nesting rule).
        segments_table_off = _build_int32_vector_table(b, [0, 0, 0])
        indices_table_off = _build_int32_vector_table(b, [])

        dim0_off = _build_dim_metadata_dense(b, 2)
        dim1_off = _build_dim_metadata_sparse_csr(b, segments_table_off,
                                                  indices_table_off)
        dim_metadata_off = _create_offset_vector(b, [dim0_off, dim1_off])
    elif dim_metadata_mode == "empty":
        dim_metadata_off = _create_offset_vector(b, [])
    elif dim_metadata_mode == "missing":
        dim_metadata_off = 0
    else:
        raise ValueError("dim_metadata_mode=" + repr(dim_metadata_mode))

    b.StartObject(3)
    if traversal_order_off:
        b.PrependUOffsetTRelativeSlot(0, traversal_order_off, 0)
    if block_map_off:
        b.PrependUOffsetTRelativeSlot(1, block_map_off, 0)
    if dim_metadata_mode != "missing":
        b.PrependUOffsetTRelativeSlot(2, dim_metadata_off, 0)
    return b.EndObject()


def _build_buffer(b, payload):
    """Buffer { data:[ubyte]; } - field 0."""
    if payload is None or len(payload) == 0:
        b.StartObject(1)
        return b.EndObject()
    data_off = _create_byte_vector(b, payload)
    b.StartObject(1)
    b.PrependUOffsetTRelativeSlot(0, data_off, 0)
    return b.EndObject()


def _build_tensor(b, *, shape, type_code, buffer_idx, name_off,
                  sparsity_off=0):
    """
    Tensor {
      shape:[int]; type:TensorType; buffer:uint; name:string;
      quantization:QuantizationParameters; is_variable:bool=false;
      sparsity:SparsityParameters; ...
    }
    """
    shape_off = _create_int_vector(b, shape)
    b.StartObject(10)
    b.PrependUOffsetTRelativeSlot(0, shape_off, 0)
    b.PrependInt8Slot(1, type_code, 0)
    b.PrependUint32Slot(2, buffer_idx, 0)
    b.PrependUOffsetTRelativeSlot(3, name_off, 0)
    if sparsity_off:
        b.PrependUOffsetTRelativeSlot(6, sparsity_off, 0)
    return b.EndObject()


def _build_opcode(b, builtin):
    """OperatorCode { deprecated_builtin_code:byte; ...; version:int; builtin_code:int; }.

    For backward compatibility, BuiltinCode() readers fall back to the legacy
    8-bit deprecated_builtin_code (slot 0) whenever the 32-bit builtin_code
    (slot 3) is below PLACEHOLDER_FOR_GREATER_OP_CODES (127). Both ADD and
    DEQUANTIZE are below that threshold, so the value MUST go into slot 0.
    """
    b.StartObject(4)
    b.PrependInt8Slot(0, builtin, 0)
    b.PrependInt32Slot(2, 1, 0)           # version = 1
    return b.EndObject()


def _build_add_options(b):
    """schema.fbs:AddOptions { fused_activation_function:ActivationFunctionType=NONE; pot_scale_int16:bool=true; }
    All fields take their defaults; we still need a non-null table so that
    decoder_flatbuffer.h:40's `opts != nullptr` check passes when the ADD
    translator looks up fused_activation_function.
    """
    b.StartObject(2)
    # field 0 = fused_activation_function (default NONE = 0): omitted to keep
    # default; field 1 = pot_scale_int16 (default true): omitted to keep default.
    return b.EndObject()


def _build_operator(b, opcode_index, inputs, outputs,
                    builtin_options_type=BO_NONE, builtin_options_off=0):
    inputs_off = _create_int_vector(b, inputs)
    outputs_off = _create_int_vector(b, outputs)
    b.StartObject(9)
    b.PrependUint32Slot(0, opcode_index, 0)
    b.PrependUOffsetTRelativeSlot(1, inputs_off, 0)
    b.PrependUOffsetTRelativeSlot(2, outputs_off, 0)
    if builtin_options_type != BO_NONE:
        b.PrependUint8Slot(3, builtin_options_type, 0)
        b.PrependUOffsetTRelativeSlot(4, builtin_options_off, 0)
    return b.EndObject()


def _build_subgraph(b, *, tensors, inputs, outputs, operators, name_off):
    tensors_off = _create_offset_vector(b, tensors)
    inputs_off = _create_int_vector(b, inputs)
    outputs_off = _create_int_vector(b, outputs)
    operators_off = _create_offset_vector(b, operators)
    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(0, tensors_off, 0)
    b.PrependUOffsetTRelativeSlot(1, inputs_off, 0)
    b.PrependUOffsetTRelativeSlot(2, outputs_off, 0)
    b.PrependUOffsetTRelativeSlot(3, operators_off, 0)
    b.PrependUOffsetTRelativeSlot(4, name_off, 0)
    return b.EndObject()


# ---------- Top-level builder ----------

def build_incomplete_sparsity_model(*,
                                    include_traversal_order=True,
                                    include_block_map=True,
                                    dim_metadata_mode="full",
                                    graph="add"):
    """
    Build a tiny .tflite model whose only constant tensor carries a non-null
    SparsityParameters with the requested completeness profile.

    graph:
        "add"        - a single ADD(input, const) op, all FLOAT32, shape (2, 2).
        "dequantize" - DEQUANTIZE(const) -> mid; ADD(input, mid) -> output.
                       const is INT8 (2, 2) with 4-byte buffer; everything
                       else is FLOAT32 (2, 2).
    """
    b = flatbuffers.Builder(1024)

    # -- Build leaf objects (strings, sparsity, buffers, tensors) bottom-up --

    # Strings (must be created before tables that reference them.)
    str_input = b.CreateString("input")
    str_const = b.CreateString("sparse_const")
    str_output = b.CreateString("output")
    str_main = b.CreateString("main")
    str_desc = b.CreateString("sparse_incomplete_test")
    str_mid = b.CreateString("dequant_out") if graph == "dequantize" else 0

    # Sparsity object referenced by the const tensor.
    sparsity_off = _build_sparsity_params(
        b,
        include_traversal_order=include_traversal_order,
        include_block_map=include_block_map,
        dim_metadata_mode=dim_metadata_mode,
    )

    # Const tensor's raw buffer: prod(shape) * elem_size bytes.
    if graph == "add":
        const_dtype = TT_FLOAT32
        const_payload = struct.pack('<4f', 1.0, 2.0, 3.0, 4.0)  # 16 B
    else:
        const_dtype = TT_INT8
        const_payload = struct.pack('<4b', 1, 2, 3, 4)  # 4 B

    # Buffers: index 0 must exist and be empty (TFLite convention).
    buffer0 = _build_buffer(b, b"")
    buffer1 = _build_buffer(b, const_payload)        # const tensor data
    buffer2 = _build_buffer(b, b"")                  # input placeholder
    buffer3 = _build_buffer(b, b"")                  # output placeholder
    buffers = [buffer0, buffer1, buffer2, buffer3]
    if graph == "dequantize":
        buffer4 = _build_buffer(b, b"")              # dequant intermediate
        buffers.append(buffer4)

    # Tensors.
    shape_22 = [2, 2]
    tensor_input = _build_tensor(b, shape=shape_22, type_code=TT_FLOAT32,
                                 buffer_idx=2, name_off=str_input)
    tensor_const = _build_tensor(b, shape=shape_22, type_code=const_dtype,
                                 buffer_idx=1, name_off=str_const,
                                 sparsity_off=sparsity_off)
    tensor_output = _build_tensor(b, shape=shape_22, type_code=TT_FLOAT32,
                                  buffer_idx=3, name_off=str_output)
    tensors = [tensor_input, tensor_const, tensor_output]
    if graph == "dequantize":
        tensor_mid = _build_tensor(b, shape=shape_22, type_code=TT_FLOAT32,
                                   buffer_idx=4, name_off=str_mid)
        tensors.append(tensor_mid)

    # Operator codes + Operators.
    # AddOptions table is mandatory for ADD: the translator reads
    # fused_activation_function via decoder_flatbuffer.h::get_attribute(),
    # which dereferences builtin_options_as<AddOptions>() and asserts non-null.
    if graph == "add":
        opcode_add = _build_opcode(b, OP_ADD)
        opcodes = [opcode_add]
        add_options_off = _build_add_options(b)
        operator_add = _build_operator(b, opcode_index=0,
                                       inputs=[0, 1], outputs=[2],
                                       builtin_options_type=BO_AddOptions,
                                       builtin_options_off=add_options_off)
        operators = [operator_add]
    else:
        opcode_deq = _build_opcode(b, OP_DEQUANTIZE)
        opcode_add = _build_opcode(b, OP_ADD)
        opcodes = [opcode_deq, opcode_add]
        add_options_off = _build_add_options(b)
        # opcode_index 0 = DEQUANTIZE on the const, output to tensor 3 (mid)
        operator_deq = _build_operator(b, opcode_index=0,
                                       inputs=[1], outputs=[3])
        # opcode_index 1 = ADD(input, mid) -> output
        operator_add = _build_operator(b, opcode_index=1,
                                       inputs=[0, 3], outputs=[2],
                                       builtin_options_type=BO_AddOptions,
                                       builtin_options_off=add_options_off)
        operators = [operator_deq, operator_add]

    # SubGraph.
    subgraph0 = _build_subgraph(b, tensors=tensors, inputs=[0], outputs=[2],
                                operators=operators, name_off=str_main)

    # Top-level Model.
    opcodes_off = _create_offset_vector(b, opcodes)
    subgraphs_off = _create_offset_vector(b, [subgraph0])
    buffers_off = _create_offset_vector(b, buffers)

    b.StartObject(7)
    b.PrependUint32Slot(0, 3, 0)                                   # version
    b.PrependUOffsetTRelativeSlot(1, opcodes_off, 0)               # operator_codes
    b.PrependUOffsetTRelativeSlot(2, subgraphs_off, 0)             # subgraphs
    b.PrependUOffsetTRelativeSlot(3, str_desc, 0)                  # description
    b.PrependUOffsetTRelativeSlot(4, buffers_off, 0)               # buffers
    model = b.EndObject()

    b.Finish(model, file_identifier=b"TFL3")
    return bytes(b.Output())


def main():
    if len(sys.argv) != 2:
        print("Usage: " + sys.argv[0] + " <output_dir>", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(sys.argv[1], "sparse_incomplete")
    os.makedirs(out_dir, exist_ok=True)

    cases = [
        ("missing_dim_metadata.tflite", dict(
            include_traversal_order=True,
            include_block_map=True,
            dim_metadata_mode="missing",
            graph="add")),
        ("empty_dim_metadata.tflite", dict(
            include_traversal_order=True,
            include_block_map=True,
            dim_metadata_mode="empty",
            graph="add")),
        ("missing_traversal_order.tflite", dict(
            include_traversal_order=False,
            include_block_map=True,
            dim_metadata_mode="full",
            graph="add")),
        ("missing_block_map.tflite", dict(
            include_traversal_order=True,
            include_block_map=False,
            dim_metadata_mode="full",
            graph="add")),
        ("empty_sparsity.tflite", dict(
            include_traversal_order=False,
            include_block_map=False,
            dim_metadata_mode="missing",
            graph="add")),
        ("dequantize_incomplete_sparsity.tflite", dict(
            include_traversal_order=False,
            include_block_map=False,
            dim_metadata_mode="missing",
            graph="dequantize")),
    ]

    for filename, kwargs in cases:
        data = build_incomplete_sparsity_model(**kwargs)
        with open(os.path.join(out_dir, filename), "wb") as f:
            f.write(data)


if __name__ == "__main__":
    main()
