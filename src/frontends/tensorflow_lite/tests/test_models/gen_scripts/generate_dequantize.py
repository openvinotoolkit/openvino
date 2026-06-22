# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import struct
import sys

import flatbuffers
from flatbuffers import builder as flatbuffers_builder

import tensorflow as tf

# Create the graph and model
class SampleGraph(tf.Module):
  def __init__(self):
    super(SampleGraph, self).__init__()
    self.var1 = tf.constant([[1, 0.75],[2000.43, -0.12345]], dtype=tf.float32)
  @tf.function(input_signature=[tf.TensorSpec([2,2], tf.float32)])
  def __call__(self, x):
    res = self.var1 + x
    return {'test_output_name': res}

module = SampleGraph()
sm_path = os.path.join(sys.argv[1], "dequantize")
tf.saved_model.save(module, sm_path)

converter = tf.lite.TFLiteConverter.from_saved_model(sm_path) # path to the SavedModel directory
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the model.
with open(os.path.join(sys.argv[1], sm_path + ".tflite"), 'wb') as f:
  f.write(tflite_model)


# TFLite TensorType byte enum values (from schema.fbs)
TENSOR_TYPE_FLOAT32 = 0
TENSOR_TYPE_FLOAT16 = 1
TENSOR_TYPE_UINT8   = 3
TENSOR_TYPE_INT8    = 9

# TFLite builtin opcode for DEQUANTIZE
DEQUANTIZE_OPCODE = 6


def build_dequantize_tflite(input_type, output_type, scale, zero_point, shape):
    """Build a minimal TFLite model: input_tensor -> DEQUANTIZE -> output_tensor.

    The TFLite flatbuffer schema field IDs used here:
      Tensor:             shape(0), type(1), buffer(2), name(3), quantization(4)
      QuantizationParams: min(0), max(1), scale(2), zero_point(3)
      SubGraph:           tensors(0), inputs(1), outputs(2), operators(3), name(4)
      Operator:           opcode_index(0), inputs(1), outputs(2)
      OperatorCode:       deprecated_builtin_code(0), custom_code(1), version(2), builtin_code(3)
      Model:              version(0), operator_codes(1), subgraphs(2), description(3), buffers(4)
    """
    fbb = flatbuffers_builder.Builder(512)

    # ---- output tensor (index 1) ----
    out_name = fbb.CreateString("tensor_1")
    fbb.StartVector(4, len(shape), 4)
    for dim in reversed(shape):
        fbb.PrependInt32(dim)
    out_shape = fbb.EndVector(len(shape))

    fbb.StartObject(0)
    out_quant = fbb.EndObject()

    fbb.StartObject(5)
    fbb.PrependUOffsetTRelative(out_quant); fbb.Slot(4)   # quantization
    fbb.PrependUOffsetTRelative(out_name);  fbb.Slot(3)   # name
    # buffer (slot 2) absent — default 0
    fbb.PrependByte(output_type);           fbb.Slot(1)   # type (byte enum)
    fbb.PrependUOffsetTRelative(out_shape); fbb.Slot(0)   # shape
    tensor_out = fbb.EndObject()

    # ---- input tensor (index 0) with quantization ----
    in_name = fbb.CreateString("tensor_0")
    fbb.StartVector(4, len(shape), 4)
    for dim in reversed(shape):
        fbb.PrependInt32(dim)
    in_shape = fbb.EndVector(len(shape))

    fbb.StartVector(4, 1, 4)
    fbb.PrependFloat32(scale)
    scale_vec = fbb.EndVector(1)

    fbb.StartVector(8, 1, 8)
    fbb.PrependInt64(zero_point)
    zp_vec = fbb.EndVector(1)

    fbb.StartObject(7)
    fbb.PrependUOffsetTRelative(zp_vec);    fbb.Slot(3)   # zero_point
    fbb.PrependUOffsetTRelative(scale_vec); fbb.Slot(2)   # scale
    in_quant = fbb.EndObject()

    fbb.StartObject(5)
    fbb.PrependUOffsetTRelative(in_quant);  fbb.Slot(4)   # quantization
    fbb.PrependUOffsetTRelative(in_name);   fbb.Slot(3)   # name
    # buffer (slot 2) absent — default 0
    fbb.PrependByte(input_type);            fbb.Slot(1)   # type (byte enum)
    fbb.PrependUOffsetTRelative(in_shape);  fbb.Slot(0)   # shape
    tensor_in = fbb.EndObject()

    # tensors vector: [tensor_in(0), tensor_out(1)]
    # PrependUOffsetTRelative adds backwards; last prepended → index 0
    fbb.StartVector(4, 2, 4)
    fbb.PrependUOffsetTRelative(tensor_out)
    fbb.PrependUOffsetTRelative(tensor_in)
    tensors_vec = fbb.EndVector(2)

    # ---- DEQUANTIZE operator ----
    fbb.StartVector(4, 1, 4)
    fbb.PrependInt32(0)
    op_inputs = fbb.EndVector(1)

    fbb.StartVector(4, 1, 4)
    fbb.PrependInt32(1)
    op_outputs = fbb.EndVector(1)

    # opcode_index absent (default 0 = DEQUANTIZE)
    fbb.StartObject(3)
    fbb.PrependUOffsetTRelative(op_outputs); fbb.Slot(2)
    fbb.PrependUOffsetTRelative(op_inputs);  fbb.Slot(1)
    op = fbb.EndObject()

    fbb.StartVector(4, 1, 4)
    fbb.PrependUOffsetTRelative(op)
    ops_vec = fbb.EndVector(1)

    # ---- subgraph ----
    fbb.StartVector(4, 1, 4); fbb.PrependInt32(0); sg_inputs  = fbb.EndVector(1)
    fbb.StartVector(4, 1, 4); fbb.PrependInt32(1); sg_outputs = fbb.EndVector(1)
    sg_name = fbb.CreateString("test_subgraph")

    fbb.StartObject(5)
    fbb.PrependUOffsetTRelative(sg_name);    fbb.Slot(4)
    fbb.PrependUOffsetTRelative(ops_vec);    fbb.Slot(3)
    fbb.PrependUOffsetTRelative(sg_outputs); fbb.Slot(2)
    fbb.PrependUOffsetTRelative(sg_inputs);  fbb.Slot(1)
    fbb.PrependUOffsetTRelative(tensors_vec);fbb.Slot(0)
    subgraph = fbb.EndObject()

    fbb.StartVector(4, 1, 4)
    fbb.PrependUOffsetTRelative(subgraph)
    subgraphs_vec = fbb.EndVector(1)

    # ---- OperatorCode: DEQUANTIZE ----
    fbb.StartObject(4)
    fbb.PrependInt32(DEQUANTIZE_OPCODE); fbb.Slot(3)   # builtin_code
    fbb.PrependInt32(1);                 fbb.Slot(2)   # version
    fbb.PrependByte(DEQUANTIZE_OPCODE);  fbb.Slot(0)   # deprecated_builtin_code
    op_code = fbb.EndObject()

    fbb.StartVector(4, 1, 4)
    fbb.PrependUOffsetTRelative(op_code)
    op_codes_vec = fbb.EndVector(1)

    # ---- two empty buffers ----
    fbb.StartObject(0); buf0 = fbb.EndObject()
    fbb.StartObject(0); buf1 = fbb.EndObject()
    fbb.StartVector(4, 2, 4)
    fbb.PrependUOffsetTRelative(buf1)
    fbb.PrependUOffsetTRelative(buf0)
    buffers_vec = fbb.EndVector(2)

    model_name = fbb.CreateString("test_model")

    fbb.StartObject(5)
    fbb.PrependUOffsetTRelative(buffers_vec);   fbb.Slot(4)
    fbb.PrependUOffsetTRelative(model_name);    fbb.Slot(3)
    fbb.PrependUOffsetTRelative(subgraphs_vec); fbb.Slot(2)
    fbb.PrependUOffsetTRelative(op_codes_vec);  fbb.Slot(1)
    fbb.PrependInt32(3);                        fbb.Slot(0)   # version
    model = fbb.EndObject()

    fbb.Finish(model)
    raw = bytearray(fbb.Output())

    # Patch in the TFL3 file identifier.  flatbuffers.Builder.Finish() writes a
    # plain root-offset prefix; TFLite requires a 4-byte "TFL3" magic after it.
    # Insert the identifier and adjust the root offset accordingly.
    root_off = struct.unpack_from('<I', raw, 0)[0]
    result = bytearray(4) + b'TFL3' + bytes(raw[4:])
    struct.pack_into('<I', result, 0, root_off + 4)
    return bytes(result)


# Model: (int8)[12] -> DEQUANTIZE -> (float32)[12], scale=0.25, zero_point=16
sm_path = os.path.join(sys.argv[1], "dequantize_int8")
with open(sm_path + ".tflite", 'wb') as f:
    f.write(build_dequantize_tflite(TENSOR_TYPE_INT8, TENSOR_TYPE_FLOAT32,
                                    scale=0.25, zero_point=16, shape=[12]))

# Model: (uint8)[12] -> DEQUANTIZE -> (float32)[12], scale=0.25, zero_point=16
sm_path = os.path.join(sys.argv[1], "dequantize_uint8")
with open(sm_path + ".tflite", 'wb') as f:
    f.write(build_dequantize_tflite(TENSOR_TYPE_UINT8, TENSOR_TYPE_FLOAT32,
                                    scale=0.25, zero_point=16, shape=[12]))

# Model: (int8)[12] -> DEQUANTIZE -> (float16)[12], scale=0.25, zero_point=16
# Tests fp16 dequantize path: weights stay in f16 to match fp16 activations.
sm_path = os.path.join(sys.argv[1], "dequantize_fp16")
with open(sm_path + ".tflite", 'wb') as f:
    f.write(build_dequantize_tflite(TENSOR_TYPE_INT8, TENSOR_TYPE_FLOAT16,
                                    scale=0.25, zero_point=16, shape=[12]))
