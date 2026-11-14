# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate crafted .tflite files for the TensorFlow Lite frontend decoder tests.

Each file is a structurally valid FlatBuffer (it passes tflite::VerifyModelBuffer) that exercises
a loader path which previously copied data out of bounds of a present-but-empty constant buffer or
dereferenced a null optional FlatBuffers vector:

  empty_dense_buffer.tflite        - a dense (non-sparse) INT32 constant of shape [1, 4096] whose
                                     Buffer.data vector is present but empty; the shape requires
                                     16 KiB. Before the decoder check the loader copied 16 KiB out
                                     of the zero-length buffer.
  empty_dense_buffer_large.tflite  - the same with shape [1, 262144] (1 MiB required). Before the
                                     decoder check loading this file crashed the process.
  squeeze_without_dims.tflite      - a SQUEEZE op whose SqueezeOptions table is present but whose
                                     squeeze_dims vector field is absent.
  operator_without_inputs.tflite   - an operator whose 'inputs' vector field is absent; its
                                     constant buffer is correctly sized.
  negative_input_index.tflite      - an operator whose input tensor index is a negative value
                                     other than the -1 optional-input sentinel; its constant
                                     buffer is correctly sized.

The files are built with the flatbuffers Python package and the TFLite schema object API shipped
with TensorFlow, so the encoder is upstream code, not a hand-rolled builder.
"""

import os
import sys

import flatbuffers
from tensorflow.lite.python import schema_py_generated as fb

TT = fb.TensorType
OP = fb.BuiltinOperator


def buffer(data):
    b = fb.BufferT()
    b.data = data  # None -> field absent; [] -> present but empty
    return b


def tensor(name, shape, ttype, buffer_idx):
    t = fb.TensorT()
    t.name = name
    t.shape = list(shape)
    t.type = ttype
    t.buffer = buffer_idx
    return t


def opcode(code):
    c = fb.OperatorCodeT()
    c.deprecatedBuiltinCode = code if code < 127 else 127
    c.builtinCode = code
    c.version = 1
    return c


def finish(subgraph, opcodes, buffers, desc):
    m = fb.ModelT()
    m.version = 3
    m.operatorCodes = opcodes
    m.subgraphs = [subgraph]
    m.buffers = buffers
    m.description = desc
    b = flatbuffers.Builder(1024)
    b.Finish(m.Pack(b), file_identifier=b"TFL3")
    return bytes(b.Output())


def add_model(shape, const_data, inputs=(0, 1), omit_inputs=False, desc="add"):
    """ADD(input, weight) -> output, all INT32 of the given shape; weight is the constant."""
    sg = fb.SubGraphT()
    sg.name = "main"
    sg.tensors = [tensor("input", shape, TT.INT32, 1), tensor("weight", shape, TT.INT32, 2),
                  tensor("output", shape, TT.INT32, 3)]
    sg.inputs = [0]
    sg.outputs = [2]
    op = fb.OperatorT()
    op.opcodeIndex = 0
    op.inputs = None if omit_inputs else list(inputs)
    op.outputs = [2]
    op.builtinOptionsType = fb.BuiltinOptions.AddOptions
    op.builtinOptions = fb.AddOptionsT()
    sg.operators = [op]
    return finish(sg, [opcode(OP.ADD)], [buffer(None), buffer(None), buffer(const_data), buffer(None)], desc)


def squeeze_without_dims():
    sg = fb.SubGraphT()
    sg.name = "main"
    sg.tensors = [tensor("input", [1, 3], TT.FLOAT32, 1), tensor("output", [3], TT.FLOAT32, 2)]
    sg.inputs = [0]
    sg.outputs = [1]
    op = fb.OperatorT()
    op.opcodeIndex = 0
    op.inputs = [0]
    op.outputs = [1]
    so = fb.SqueezeOptionsT()
    so.squeezeDims = None  # field absent
    op.builtinOptionsType = fb.BuiltinOptions.SqueezeOptions
    op.builtinOptions = so
    sg.operators = [op]
    return finish(sg, [opcode(OP.SQUEEZE)], [buffer(None), buffer(None), buffer(None)], "squeeze")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)
    out = os.path.join(sys.argv[1], "decoder_null_and_empty_buffer")
    os.makedirs(out, exist_ok=True)

    sized_weight = [0] * 16  # 4 x INT32 for the [1, 4] probe models
    files = {
        "empty_dense_buffer.tflite": add_model([1, 4096], [], desc="empty_dense_buffer"),
        "empty_dense_buffer_large.tflite": add_model([1, 262144], [], desc="empty_dense_buffer_large"),
        "squeeze_without_dims.tflite": squeeze_without_dims(),
        "operator_without_inputs.tflite": add_model([1, 4], sized_weight, omit_inputs=True, desc="no_inputs"),
        "negative_input_index.tflite": add_model([1, 4], sized_weight, inputs=(-2, 1), desc="negative_index"),
    }
    for name, data in files.items():
        with open(os.path.join(out, name), "wb") as f:
            f.write(data)


if __name__ == "__main__":
    main()
