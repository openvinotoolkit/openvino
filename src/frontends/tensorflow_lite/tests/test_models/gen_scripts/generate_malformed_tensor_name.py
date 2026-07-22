# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Generate a crafted .tflite model whose Tensor.name field is absent.
#
# In the FlatBuffer schema for TensorFlow Lite, every table field is implicitly
# optional - the field is absent when its vtable slot offset is 0. The schema-
# generated `Tensor::name()` accessor returns nullptr for an absent name, and
# any caller that dereferences that pointer (e.g. `tensor->name()->str()`)
# crashes with a null-pointer dereference.
#
# This generator produces a structurally valid TFLite buffer
# (`tflite::VerifyModelBuffer` returns true) where one tensor has no `name`
# slot in its vtable. It is the regression input for the load-time null-name
# check in src/frontends/tensorflow_lite/src/decoder_flatbuffer.cpp.
#
# Models generated:
#   malformed_tensor_name/missing_tensor_name.tflite
#       - 2 tensors, both legal indices, but tensor 0 has no `name` field set.
#         The frontend must throw a clear ov::Exception during load(), not
#         segfault.

import os
import sys

import flatbuffers


def build_tflite_with_unnamed_tensor(unnamed_tensor_indices=()):
    """Build a minimal valid TFLite FlatBuffer with selected tensors lacking a `name`.

    Default valid model structure:
      - 2 tensors (index 0 = input, index 1 = output)
      - 3 buffers (0 = empty sentinel, 1 for tensor0, 2 for tensor1)
      - 1 operator code (ADD)
      - 1 operator: inputs=[0], outputs=[1], opcode_index=0
      - SubGraph inputs=[0], outputs=[1]

    For each tensor index in `unnamed_tensor_indices`, the `name` slot is omitted
    from the tensor's vtable so that `Tensor::name()` returns nullptr at load time.
    """
    builder = flatbuffers.Builder(1024)

    num_tensors = 2
    num_buffers = 3
    num_opcodes = 1

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
    # Pre-create name strings for the tensors that should be named. Strings are
    # built up-front because builder offsets are LIFO; we cannot create strings
    # while a table object is open.
    tensor_names = {}
    for i in range(num_tensors):
        if i not in unnamed_tensor_indices:
            tensor_names[i] = builder.CreateString(f"tensor_{i}")

    shape_vecs = []
    for _ in range(num_tensors):
        builder.StartVector(4, 1, 4)
        builder.PrependInt32(1)
        shape_vecs.append(builder.EndVector())

    tensor_offsets = []
    for i in range(num_tensors):
        buf_idx = i + 1 if i + 1 < num_buffers else 0
        builder.StartObject(11)  # Tensor: shape, type, buffer, name, ...
        builder.PrependUOffsetTRelativeSlot(0, shape_vecs[i], 0)  # shape
        builder.PrependInt8Slot(1, 0, 0)    # type = FLOAT32
        builder.PrependUint32Slot(2, buf_idx, 0)  # buffer index
        if i in tensor_names:
            builder.PrependUOffsetTRelativeSlot(3, tensor_names[i], 0)  # name
        # else: deliberately omit slot 3 - this is the regression-under-test condition.
        t_offset = builder.EndObject()
        tensor_offsets.append(t_offset)

    builder.StartVector(4, len(tensor_offsets), 4)
    for off in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(off)
    tensors_vec = builder.EndVector()

    # -- Operator --
    builder.StartVector(4, 1, 4)
    builder.PrependInt32(0)
    op_inputs_vec = builder.EndVector()

    builder.StartVector(4, 1, 4)
    builder.PrependInt32(1)
    op_outputs_vec = builder.EndVector()

    builder.StartObject(11)  # Operator: opcode_index, inputs, outputs, ...
    builder.PrependUint32Slot(0, 0, 0)
    builder.PrependUOffsetTRelativeSlot(1, op_inputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, op_outputs_vec, 0)
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

    desc = builder.CreateString("missing_tensor_name_test_model")

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
        print(f"Usage: {sys.argv[0]} <output_directory>", file=sys.stderr)
        sys.exit(1)

    path_to_model_dir = os.path.join(sys.argv[1], "malformed_tensor_name")
    os.makedirs(path_to_model_dir, exist_ok=True)

    # Tensor 0 (the graph input) has no `name` field in its vtable. The frontend
    # must reject this with an ov::Exception during load(), not crash.
    model = build_tflite_with_unnamed_tensor(unnamed_tensor_indices=(0,))
    with open(os.path.join(path_to_model_dir, "missing_tensor_name.tflite"), "wb") as f:
        f.write(model)
