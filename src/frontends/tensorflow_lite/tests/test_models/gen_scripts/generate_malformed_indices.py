# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Generate crafted .tflite model files with out-of-bounds tensor/buffer/opcode indices.
# These models test that the TFLite frontend properly validates all FlatBuffer
# vector indices before use, preventing CWE-125 (Out-of-bounds Read).
#
# Each model is a minimal valid TFLite structure with one intentionally invalid
# index that would cause an out-of-bounds read without proper bounds checking.
#
# Models generated:
#   malformed_indices/oob_output_tensor_index.tflite  - operator output index > tensors count
#   malformed_indices/oob_input_tensor_index.tflite   - operator input index > tensors count
#   malformed_indices/oob_opcode_index.tflite         - opcode index > operator_codes count
#   malformed_indices/oob_graph_io_tensor_index.tflite - graph I/O tensor index > tensors count
#   malformed_indices/oob_buffer_index.tflite         - tensor buffer index > buffers count

import os
import sys

import flatbuffers


def build_minimal_tflite(output_tensor_indices=None,
                         input_tensor_indices=None,
                         num_tensors=2,
                         num_buffers=3,
                         opcode_index=0,
                         num_opcodes=1,
                         graph_input_ids=None,
                         graph_output_ids=None,
                         tensor_buffer_indices=None):
    """
    Build a minimal .tflite FlatBuffer model using flatbuffers.Builder.

    Default valid model structure:
      - 2 tensors (index 0 = input, index 1 = output)
      - 3 buffers (0 = empty sentinel, 1 for tensor0, 2 for tensor1)
      - 1 operator code (ADD)
      - 1 operator: inputs=[0], outputs=[1], opcode_index=0
      - SubGraph inputs=[0], outputs=[1]

    Parameters allow overriding indices to produce malformed models.
    """
    builder = flatbuffers.Builder(1024)

    # -- Buffers --
    buffer_offsets = []
    for _ in range(num_buffers):
        builder.StartObject(3)  # Buffer: data, offset, size
        buf_offset = builder.EndObject()
        buffer_offsets.append(buf_offset)

    builder.StartVector(4, len(buffer_offsets), 4)
    for off in reversed(buffer_offsets):
        builder.PrependUOffsetTRelative(off)
    buffers_vec = builder.EndVector()

    # -- Operator Codes --
    opcode_offsets = []
    for _ in range(num_opcodes):
        builder.StartObject(4)  # OperatorCode: deprecated_builtin_code, custom_code, version, builtin_code
        builder.PrependInt8Slot(0, 0, 0)    # deprecated_builtin_code = ADD(0)
        builder.PrependInt32Slot(3, 0, 0)   # builtin_code = ADD(0)
        builder.PrependInt32Slot(2, 1, 1)   # version = 1
        oc_offset = builder.EndObject()
        opcode_offsets.append(oc_offset)

    builder.StartVector(4, len(opcode_offsets), 4)
    for off in reversed(opcode_offsets):
        builder.PrependUOffsetTRelative(off)
    opcodes_vec = builder.EndVector()

    # -- Tensors --
    tensor_names = []
    for i in range(num_tensors):
        tensor_names.append(builder.CreateString(f"tensor_{i}"))

    shape_vecs = []
    for _ in range(num_tensors):
        builder.StartVector(4, 1, 4)
        builder.PrependInt32(1)
        shape_vecs.append(builder.EndVector())

    tensor_offsets = []
    for i in range(num_tensors):
        buf_idx = i + 1 if i + 1 < num_buffers else 0
        if tensor_buffer_indices is not None and i < len(tensor_buffer_indices):
            buf_idx = tensor_buffer_indices[i]
        builder.StartObject(11)  # Tensor: shape, type, buffer, name, ...
        builder.PrependUOffsetTRelativeSlot(0, shape_vecs[i], 0)  # shape
        builder.PrependInt8Slot(1, 0, 0)    # type = FLOAT32
        builder.PrependUint32Slot(2, buf_idx, 0)  # buffer index
        builder.PrependUOffsetTRelativeSlot(3, tensor_names[i], 0)  # name
        t_offset = builder.EndObject()
        tensor_offsets.append(t_offset)

    builder.StartVector(4, len(tensor_offsets), 4)
    for off in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(off)
    tensors_vec = builder.EndVector()

    # -- Operator --
    op_inputs = input_tensor_indices if input_tensor_indices is not None else [0]
    builder.StartVector(4, len(op_inputs), 4)
    for idx in reversed(op_inputs):
        builder.PrependInt32(idx)
    op_inputs_vec = builder.EndVector()

    op_outputs = output_tensor_indices if output_tensor_indices is not None else [1]
    builder.StartVector(4, len(op_outputs), 4)
    for idx in reversed(op_outputs):
        builder.PrependInt32(idx)
    op_outputs_vec = builder.EndVector()

    builder.StartObject(11)  # Operator: opcode_index, inputs, outputs, ...
    builder.PrependUint32Slot(0, opcode_index, 0)
    builder.PrependUOffsetTRelativeSlot(1, op_inputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, op_outputs_vec, 0)
    op_offset = builder.EndObject()

    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(op_offset)
    operators_vec = builder.EndVector()

    # -- SubGraph --
    sg_inputs = graph_input_ids if graph_input_ids is not None else [0]
    builder.StartVector(4, len(sg_inputs), 4)
    for idx in reversed(sg_inputs):
        builder.PrependInt32(idx)
    sg_inputs_vec = builder.EndVector()

    sg_outputs = graph_output_ids if graph_output_ids is not None else [1]
    builder.StartVector(4, len(sg_outputs), 4)
    for idx in reversed(sg_outputs):
        builder.PrependInt32(idx)
    sg_outputs_vec = builder.EndVector()

    sg_name = builder.CreateString("main")

    builder.StartObject(7)  # SubGraph: tensors, inputs, outputs, operators, name, ...
    builder.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)
    builder.PrependUOffsetTRelativeSlot(1, sg_inputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, sg_outputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(3, operators_vec, 0)
    builder.PrependUOffsetTRelativeSlot(4, sg_name, 0)
    sg_offset = builder.EndObject()

    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(sg_offset)
    subgraphs_vec = builder.EndVector()

    desc = builder.CreateString("malformed_test_model")

    # -- Model --
    builder.StartObject(8)  # Model: version, operator_codes, subgraphs, description, buffers, ...
    builder.PrependUint32Slot(0, 3, 0)  # version = 3
    builder.PrependUOffsetTRelativeSlot(1, opcodes_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, subgraphs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(3, desc, 0)
    builder.PrependUOffsetTRelativeSlot(4, buffers_vec, 0)
    model_offset = builder.EndObject()

    builder.Finish(model_offset, b"TFL3")
    return bytes(builder.Output())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_directory>")
        sys.exit(1)

    path_to_model_dir = os.path.join(sys.argv[1], "malformed_indices")
    os.makedirs(path_to_model_dir, exist_ok=True)

    # 1. Operator output tensor index out of bounds
    # 2 tensors defined, operator output references index 9999
    model = build_minimal_tflite(output_tensor_indices=[9999])
    with open(os.path.join(path_to_model_dir, 'oob_output_tensor_index.tflite'), 'wb') as f:
        f.write(model)

    # 2. Operator input tensor index out of bounds
    # 2 tensors defined, operator input references index 9999
    model = build_minimal_tflite(input_tensor_indices=[9999])
    with open(os.path.join(path_to_model_dir, 'oob_input_tensor_index.tflite'), 'wb') as f:
        f.write(model)

    # 3. Operator opcode index out of bounds
    # 1 opcode defined, operator references opcode index 99
    model = build_minimal_tflite(opcode_index=99)
    with open(os.path.join(path_to_model_dir, 'oob_opcode_index.tflite'), 'wb') as f:
        f.write(model)

    # 4. Graph input/output tensor index out of bounds
    # 2 tensors defined, graph inputs/outputs reference index 9999
    model = build_minimal_tflite(graph_input_ids=[9999], graph_output_ids=[9999])
    with open(os.path.join(path_to_model_dir, 'oob_graph_io_tensor_index.tflite'), 'wb') as f:
        f.write(model)

    # 5. Tensor buffer index out of bounds
    # 3 buffers defined, tensor 1 references buffer index 9999
    model = build_minimal_tflite(tensor_buffer_indices=[1, 9999])
    with open(os.path.join(path_to_model_dir, 'oob_buffer_index.tflite'), 'wb') as f:
        f.write(model)
