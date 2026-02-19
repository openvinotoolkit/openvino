# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import struct
import sys


class FlatBufferBuilder:
    """Minimal FlatBuffer builder for crafting test .tflite files."""

    def __init__(self, initial_size=8192):
        self.buf = bytearray(initial_size)
        self.head = initial_size
        self.vtables = []
        self.object_start = 0
        self.vtable_fields = []

    def prep(self, size, additional_bytes=0):
        align_size = ((~(self.head - additional_bytes)) + 1) & (size - 1)
        total = align_size + size + additional_bytes
        while self.head < total:
            old_buf = self.buf
            self.buf = bytearray(len(self.buf) * 2)
            self.buf[len(self.buf) - len(old_buf):] = old_buf
            self.head += len(self.buf) // 2
        self.head -= align_size

    def place(self, val, fmt):
        sz = struct.calcsize(fmt)
        self.head -= sz
        struct.pack_into(fmt, self.buf, self.head, val)

    def offset(self):
        return len(self.buf) - self.head

    def create_vector(self, fmt, data):
        elem_size = struct.calcsize(fmt)
        self.prep(4, len(data) * elem_size)
        self.prep(elem_size, len(data) * elem_size)
        for val in reversed(data):
            self.head -= elem_size
            struct.pack_into(fmt, self.buf, self.head, val)
        self.prep(4)
        self.head -= 4
        struct.pack_into('<I', self.buf, self.head, len(data))
        return self.offset()

    def create_string(self, s):
        encoded = s.encode('utf-8')
        self.prep(4, len(encoded) + 1)
        self.head -= 1
        self.buf[self.head] = 0
        self.head -= len(encoded)
        self.buf[self.head:self.head + len(encoded)] = encoded
        self.prep(4)
        self.head -= 4
        struct.pack_into('<I', self.buf, self.head, len(encoded))
        return self.offset()

    def create_offset_vector(self, offsets):
        self.prep(4, 4 * len(offsets))
        for off in reversed(offsets):
            self.prep(4)
            self.head -= 4
            struct.pack_into('<I', self.buf, self.head, self.offset() - off)
        self.prep(4)
        self.head -= 4
        struct.pack_into('<I', self.buf, self.head, len(offsets))
        return self.offset()

    def start_object(self, num_fields):
        self.object_start = self.offset()
        self.vtable_fields = [0] * num_fields

    def add_field_int8(self, field_idx, val):
        self.prep(1)
        self.place(val, '<b')
        self.vtable_fields[field_idx] = self.offset()

    def add_field_uint32(self, field_idx, val):
        self.prep(4)
        self.place(val, '<I')
        self.vtable_fields[field_idx] = self.offset()

    def add_field_int32(self, field_idx, val):
        self.prep(4)
        self.place(val, '<i')
        self.vtable_fields[field_idx] = self.offset()

    def add_field_ref(self, field_idx, ref_offset):
        if ref_offset == 0:
            return
        self.prep(4)
        self.head -= 4
        struct.pack_into('<I', self.buf, self.head, self.offset() - ref_offset)
        self.vtable_fields[field_idx] = self.offset()

    def end_object(self):
        self.prep(4)
        self.head -= 4
        object_offset = self.offset()
        vtable_size = 4 + 2 * len(self.vtable_fields)
        vtable = bytearray(vtable_size)
        struct.pack_into('<H', vtable, 0, vtable_size)
        struct.pack_into('<H', vtable, 2, object_offset - self.object_start)
        for i, field_off in enumerate(self.vtable_fields):
            if field_off:
                struct.pack_into('<H', vtable, 4 + 2 * i,
                                 object_offset - field_off)

        existing_vtable = None
        for vt_off in self.vtables:
            vt_start = len(self.buf) - vt_off
            vt_len = struct.unpack_from('<H', self.buf, vt_start)[0]
            if vt_len == vtable_size and self.buf[vt_start:vt_start + vt_len] == vtable:
                existing_vtable = vt_off
                break

        if existing_vtable is not None:
            struct.pack_into('<i', self.buf, len(self.buf) - object_offset,
                             existing_vtable - object_offset)
        else:
            self.head -= len(vtable)
            self.buf[self.head:self.head + len(vtable)] = vtable
            vt_offset = self.offset()
            self.vtables.append(vt_offset)
            struct.pack_into('<i', self.buf, len(self.buf) - object_offset,
                             vt_offset - object_offset)
        return object_offset

    def finish(self, root_offset, file_identifier=None):
        if file_identifier:
            self.prep(4, 4 + 4)
            ident = file_identifier.encode('ascii')
            for i in range(3, -1, -1):
                self.head -= 1
                self.buf[self.head] = ident[i]
        else:
            self.prep(4, 4)
        self.head -= 4
        struct.pack_into('<I', self.buf, self.head, self.offset() - root_offset)

    def output(self):
        return bytes(self.buf[self.head:])


def build_undersized_buffer_model():
    """Build a .tflite with a constant tensor whose shape requires more data than buffer provides.
    Tensor shape [2,3] FLOAT32 requires 24 bytes, but buffer contains only 4 bytes."""
    b = FlatBufferBuilder(4096)

    str_input = b.create_string("input")
    str_weight = b.create_string("weight")
    str_output = b.create_string("output")

    # Buffer 0: empty (TFLite convention)
    b.start_object(1)
    buffer0 = b.end_object()

    # Buffer 1: empty (input runtime)
    b.start_object(1)
    buffer1 = b.end_object()

    # Buffer 2: weight data - only 4 bytes but shape requires 24 bytes
    weight_data = b.create_vector('<B', [0x00, 0x00, 0x80, 0x3F])  # 1.0f
    b.start_object(1)
    b.add_field_ref(0, weight_data)
    buffer2 = b.end_object()

    # Buffer 3: empty (output)
    b.start_object(1)
    buffer3 = b.end_object()

    # Tensor 0: input [2,3] f32
    shape0 = b.create_vector('<i', [2, 3])
    b.start_object(10)
    b.add_field_ref(0, shape0)
    b.add_field_int8(1, 0)    # FLOAT32
    b.add_field_uint32(2, 1)  # buffer 1 (empty = runtime)
    b.add_field_ref(3, str_input)
    tensor0 = b.end_object()

    # Tensor 1: weight [2,3] f32 - constant with undersized buffer
    shape1 = b.create_vector('<i', [2, 3])
    b.start_object(10)
    b.add_field_ref(0, shape1)
    b.add_field_int8(1, 0)    # FLOAT32
    b.add_field_uint32(2, 2)  # buffer 2 (only 4 bytes!)
    b.add_field_ref(3, str_weight)
    tensor1 = b.end_object()

    # Tensor 2: output [2,3] f32
    shape2 = b.create_vector('<i', [2, 3])
    b.start_object(10)
    b.add_field_ref(0, shape2)
    b.add_field_int8(1, 0)    # FLOAT32
    b.add_field_uint32(2, 3)  # buffer 3
    b.add_field_ref(3, str_output)
    tensor2 = b.end_object()

    # OperatorCode: ADD (builtin_code=0)
    b.start_object(4)
    b.add_field_int8(0, 0)     # deprecated_builtin_code = ADD
    b.add_field_int32(2, 1)    # version
    b.add_field_int32(3, 0)    # builtin_code = ADD
    opcode = b.end_object()

    # Operator: ADD(input, weight) -> output
    op_inputs = b.create_vector('<i', [0, 1])
    op_outputs = b.create_vector('<i', [2])
    b.start_object(9)
    b.add_field_uint32(0, 0)
    b.add_field_ref(1, op_inputs)
    b.add_field_ref(2, op_outputs)
    operator0 = b.end_object()

    # SubGraph - only tensor 0 is input, tensor 1 is constant
    tensors_vec = b.create_offset_vector([tensor0, tensor1, tensor2])
    operators_vec = b.create_offset_vector([operator0])
    sg_inputs = b.create_vector('<i', [0])
    sg_outputs = b.create_vector('<i', [2])
    sg_name = b.create_string("main")

    b.start_object(5)
    b.add_field_ref(0, tensors_vec)
    b.add_field_ref(1, sg_inputs)
    b.add_field_ref(2, sg_outputs)
    b.add_field_ref(3, operators_vec)
    b.add_field_ref(4, sg_name)
    subgraph0 = b.end_object()

    # Model
    opcodes_vec = b.create_offset_vector([opcode])
    subgraphs_vec = b.create_offset_vector([subgraph0])
    buffers_vec = b.create_offset_vector([buffer0, buffer1, buffer2, buffer3])
    desc = b.create_string("undersized_buffer_test")

    b.start_object(5)
    b.add_field_uint32(0, 3)
    b.add_field_ref(1, opcodes_vec)
    b.add_field_ref(2, subgraphs_vec)
    b.add_field_ref(3, desc)
    b.add_field_ref(4, buffers_vec)
    model = b.end_object()

    b.finish(model, "TFL3")
    return b.output()


def build_empty_buffer_nonempty_shape_model():
    """Build a .tflite with a constant tensor whose shape is [2,2] but buffer data is empty."""
    b = FlatBufferBuilder(4096)

    str_input = b.create_string("input")
    str_weight = b.create_string("weight")
    str_output = b.create_string("output")

    # Buffer 0: empty (TFLite convention)
    b.start_object(1)
    buffer0 = b.end_object()

    # Buffer 1: empty (input runtime)
    b.start_object(1)
    buffer1 = b.end_object()

    # Buffer 2: weight data - 1 byte but shape requires 16 bytes
    weight_data = b.create_vector('<B', [0x01])
    b.start_object(1)
    b.add_field_ref(0, weight_data)
    buffer2 = b.end_object()

    # Buffer 3: empty (output)
    b.start_object(1)
    buffer3 = b.end_object()

    # Tensor 0: input [2,2] f32
    shape0 = b.create_vector('<i', [2, 2])
    b.start_object(10)
    b.add_field_ref(0, shape0)
    b.add_field_int8(1, 0)    # FLOAT32
    b.add_field_uint32(2, 1)  # buffer 1
    b.add_field_ref(3, str_input)
    tensor0 = b.end_object()

    # Tensor 1: weight [2,2] f32 - constant with 1-byte buffer (needs 16 bytes)
    shape1 = b.create_vector('<i', [2, 2])
    b.start_object(10)
    b.add_field_ref(0, shape1)
    b.add_field_int8(1, 0)    # FLOAT32
    b.add_field_uint32(2, 2)  # buffer 2 (only 1 byte!)
    b.add_field_ref(3, str_weight)
    tensor1 = b.end_object()

    # Tensor 2: output [2,2] f32
    shape2 = b.create_vector('<i', [2, 2])
    b.start_object(10)
    b.add_field_ref(0, shape2)
    b.add_field_int8(1, 0)    # FLOAT32
    b.add_field_uint32(2, 3)  # buffer 3
    b.add_field_ref(3, str_output)
    tensor2 = b.end_object()

    # OperatorCode: ADD
    b.start_object(4)
    b.add_field_int8(0, 0)
    b.add_field_int32(2, 1)
    b.add_field_int32(3, 0)
    opcode = b.end_object()

    # Operator: ADD(input, weight) -> output
    op_inputs = b.create_vector('<i', [0, 1])
    op_outputs = b.create_vector('<i', [2])
    b.start_object(9)
    b.add_field_uint32(0, 0)
    b.add_field_ref(1, op_inputs)
    b.add_field_ref(2, op_outputs)
    operator0 = b.end_object()

    # SubGraph
    tensors_vec = b.create_offset_vector([tensor0, tensor1, tensor2])
    operators_vec = b.create_offset_vector([operator0])
    sg_inputs = b.create_vector('<i', [0])
    sg_outputs = b.create_vector('<i', [2])
    sg_name = b.create_string("main")

    b.start_object(5)
    b.add_field_ref(0, tensors_vec)
    b.add_field_ref(1, sg_inputs)
    b.add_field_ref(2, sg_outputs)
    b.add_field_ref(3, operators_vec)
    b.add_field_ref(4, sg_name)
    subgraph0 = b.end_object()

    # Model
    opcodes_vec = b.create_offset_vector([opcode])
    subgraphs_vec = b.create_offset_vector([subgraph0])
    buffers_vec = b.create_offset_vector([buffer0, buffer1, buffer2, buffer3])
    desc = b.create_string("empty_buffer_nonempty_shape_test")

    b.start_object(5)
    b.add_field_uint32(0, 3)
    b.add_field_ref(1, opcodes_vec)
    b.add_field_ref(2, subgraphs_vec)
    b.add_field_ref(3, desc)
    b.add_field_ref(4, buffers_vec)
    model = b.end_object()

    b.finish(model, "TFL3")
    return b.output()


path_to_model_dir = os.path.join(sys.argv[1], "bad_buffer_size")
os.makedirs(path_to_model_dir, exist_ok=True)

data = build_undersized_buffer_model()
with open(os.path.join(path_to_model_dir, 'undersized_buffer.tflite'), 'wb') as f:
    f.write(data)

data = build_empty_buffer_nonempty_shape_model()
with open(os.path.join(path_to_model_dir, 'empty_buffer_nonempty_shape.tflite'), 'wb') as f:
    f.write(data)
