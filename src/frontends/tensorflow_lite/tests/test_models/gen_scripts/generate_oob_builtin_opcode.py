# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# SDL505 fuzzing finding: generate minimal .tflite models with out-of-bounds
# builtin operator codes to test that the frontend rejects them cleanly (CWE-125).
#
# oob_deprecated_builtin_code.tflite — deprecated_builtin_code=-1 (0xFF as ubyte,
#   read back as int8_t=-1 < BuiltinOperator_MIN=0; triggers lower-bound check)
# oob_builtin_code.tflite — builtin_code=9999 > BuiltinOperator_MAX=209
# null_sentinel_builtin_code.tflite — builtin_code=210, the nullptr entry
#   at EnumNamesBuiltinOperator()[210] (one past STABLEHLO_CASE=209=MAX)

import os
import sys

import flatbuffers


def build_minimal_tflite_with_opcode(deprecated_builtin_code, builtin_code):
    """
    Build a minimal .tflite FlatBuffer with one operator whose operator code
    fields are set to the supplied values.  Everything else is a valid minimal
    model so that FlatBuffer verification passes and the frontend reaches the
    opcode lookup path.
    """
    builder = flatbuffers.Builder(1024)

    # Buffers (3: sentinel + one per tensor)
    buffer_offsets = []
    for _ in range(3):
        builder.StartObject(3)
        buffer_offsets.append(builder.EndObject())
    builder.StartVector(4, 3, 4)
    for off in reversed(buffer_offsets):
        builder.PrependUOffsetTRelative(off)
    buffers_vec = builder.EndVector()

    # OperatorCode with the supplied (potentially invalid) code values
    builder.StartObject(4)  # deprecated_builtin_code, custom_code, version, builtin_code
    builder.PrependInt8Slot(0, deprecated_builtin_code, 0)
    builder.PrependInt32Slot(3, builtin_code, 0)
    builder.PrependInt32Slot(2, 1, 1)   # version = 1
    oc_offset = builder.EndObject()
    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(oc_offset)
    opcodes_vec = builder.EndVector()

    # Tensors (input + output)
    tensor_names = [builder.CreateString(f"tensor_{i}") for i in range(2)]
    shape_vecs = []
    for _ in range(2):
        builder.StartVector(4, 1, 4)
        builder.PrependInt32(1)
        shape_vecs.append(builder.EndVector())
    tensor_offsets = []
    for i in range(2):
        builder.StartObject(11)
        builder.PrependUOffsetTRelativeSlot(0, shape_vecs[i], 0)
        builder.PrependInt8Slot(1, 0, 0)       # type = FLOAT32
        builder.PrependUint32Slot(2, i + 1, 0) # buffer index
        builder.PrependUOffsetTRelativeSlot(3, tensor_names[i], 0)
        tensor_offsets.append(builder.EndObject())
    builder.StartVector(4, 2, 4)
    for off in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(off)
    tensors_vec = builder.EndVector()

    # Operator: inputs=[0], outputs=[1], opcode_index=0
    builder.StartVector(4, 1, 4)
    builder.PrependInt32(0)
    op_inputs_vec = builder.EndVector()
    builder.StartVector(4, 1, 4)
    builder.PrependInt32(1)
    op_outputs_vec = builder.EndVector()
    builder.StartObject(11)
    builder.PrependUint32Slot(0, 0, 0)
    builder.PrependUOffsetTRelativeSlot(1, op_inputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, op_outputs_vec, 0)
    op_offset = builder.EndObject()
    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(op_offset)
    operators_vec = builder.EndVector()

    # SubGraph
    builder.StartVector(4, 1, 4)
    builder.PrependInt32(0)
    sg_inputs_vec = builder.EndVector()
    builder.StartVector(4, 1, 4)
    builder.PrependInt32(1)
    sg_outputs_vec = builder.EndVector()
    sg_name = builder.CreateString("main")
    builder.StartObject(7)
    builder.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)
    builder.PrependUOffsetTRelativeSlot(1, sg_inputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(2, sg_outputs_vec, 0)
    builder.PrependUOffsetTRelativeSlot(3, operators_vec, 0)
    builder.PrependUOffsetTRelativeSlot(4, sg_name, 0)
    sg_offset = builder.EndObject()
    builder.StartVector(4, 1, 4)
    builder.PrependUOffsetTRelative(sg_offset)
    subgraphs_vec = builder.EndVector()

    desc = builder.CreateString("oob_opcode_test_model")

    # Model
    builder.StartObject(8)
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

    path_to_model_dir = os.path.join(sys.argv[1], "oob_builtin_opcode")
    os.makedirs(path_to_model_dir, exist_ok=True)

    # 1. deprecated_builtin_code path: value is stored as 0xFF in the flatbuffer
    #    ubyte field, which C++ reads back as int8_t = -1. That is below
    #    BuiltinOperator_MIN (0), triggering the lower-bound check.
    #    Without the fix: EnumNamesBuiltinOperator()[-1] → read before array start.
    model = build_minimal_tflite_with_opcode(deprecated_builtin_code=-1, builtin_code=0)
    with open(os.path.join(path_to_model_dir, 'oob_deprecated_builtin_code.tflite'), 'wb') as f:
        f.write(model)

    # 2. builtin_code path: deprecated_builtin_code = 127 (>= PLACEHOLDER threshold)
    #    so the builtin_code field is used instead; set it to 9999 > BuiltinOperator_MAX (209).
    model = build_minimal_tflite_with_opcode(deprecated_builtin_code=127, builtin_code=9999)
    with open(os.path.join(path_to_model_dir, 'oob_builtin_code.tflite'), 'wb') as f:
        f.write(model)

    # 3. Second fuzzing crash: builtin_code = 210, which is the
    #    nullptr sentinel at EnumNamesBuiltinOperator()[210] (one past STABLEHLO_CASE=209=MAX).
    #    Without the fix: string::operator=(nullptr) → strlen(nullptr) → SIGSEGV.
    model = build_minimal_tflite_with_opcode(deprecated_builtin_code=127, builtin_code=210)
    with open(os.path.join(path_to_model_dir, 'null_sentinel_builtin_code.tflite'), 'wb') as f:
        f.write(model)
