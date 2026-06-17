# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Generates malicious .tflite files for testing sparse tensor index validation.
# These files are structurally valid FlatBuffers but contain semantically invalid
# sparse index values that should be rejected during deserialization.
#
# Test models generated:
# 1. sparse_oob_index.tflite - sparse tensor with index value 999 in a dimension of size 100
#    (triggers CWE-787 without fix: heap OOB write at row_offset = 999 * 4 = 3996 in 800-byte buffer)
# 2. sparse_negative_index.tflite - sparse tensor with negative index value (-1)
# 3. sparse_non_monotonic_segments.tflite - sparse tensor with non-monotonic segments

import os
import struct
import sys


class FlatBufferBuilder:
    """Minimal FlatBuffer builder for crafting test .tflite files with sparse tensors."""

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

    def add_field_uint8(self, field_idx, val):
        self.prep(1)
        self.place(val, '<B')
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


def build_sparse_model(indices, segments, shape=(2, 100), num_values=3):
    """Build a minimal .tflite file with a sparse tensor using the given indices and segments.

    Args:
        indices: list of int32 index values for the sparse dimension
        segments: list of int32 segment values for CSR format
        shape: tensor shape (default [2, 100])
        num_values: number of float32 sparse values to include
    """
    b = FlatBufferBuilder(8192)

    sparse_values = struct.pack('<' + 'f' * num_values, *([1.0] * num_values))

    str_input = b.create_string("input")
    str_weight = b.create_string("sparse_weight")
    str_output = b.create_string("output")

    # Int32Vector tables for segments and indices
    # Segments
    segments_values = b.create_vector('<i', segments)
    b.start_object(1)
    b.add_field_ref(0, segments_values)
    segments_table = b.end_object()

    # Indices
    indices_values = b.create_vector('<i', indices)
    b.start_object(1)
    b.add_field_ref(0, indices_values)
    indices_table = b.end_object()

    # DimensionMetadata[0]: DENSE, dense_size = shape[0]
    b.start_object(6)
    b.add_field_int8(0, 0)                # format = DENSE
    b.add_field_int32(1, shape[0])        # dense_size
    b.add_field_uint8(2, 0)              # segments_type = NONE
    b.add_field_uint8(4, 0)              # indices_type = NONE
    dim0_meta = b.end_object()

    # DimensionMetadata[1]: SPARSE_CSR with Int32Vector segments/indices
    b.start_object(6)
    b.add_field_int8(0, 1)               # format = SPARSE_CSR
    b.add_field_int32(1, 0)              # dense_size = 0 (unused for sparse)
    b.add_field_uint8(2, 1)              # segments_type = Int32Vector
    b.add_field_ref(3, segments_table)
    b.add_field_uint8(4, 1)              # indices_type = Int32Vector
    b.add_field_ref(5, indices_table)
    dim1_meta = b.end_object()

    dim_metadata_vec = b.create_offset_vector([dim0_meta, dim1_meta])
    traversal_order_vec = b.create_vector('<i', [0, 1])
    block_map_vec = b.create_vector('<i', [0])

    # SparsityParameters
    b.start_object(3)
    b.add_field_ref(0, traversal_order_vec)
    b.add_field_ref(1, block_map_vec)
    b.add_field_ref(2, dim_metadata_vec)
    sparsity_params = b.end_object()

    # Buffers
    b.start_object(1)
    buffer0 = b.end_object()

    buf1_data = b.create_vector('<B', list(sparse_values))
    b.start_object(1)
    b.add_field_ref(0, buf1_data)
    buffer1 = b.end_object()

    b.start_object(1)
    buffer2 = b.end_object()

    b.start_object(1)
    buffer3 = b.end_object()

    # Tensors
    shape_input = b.create_vector('<i', [1, shape[1]])
    b.start_object(10)
    b.add_field_ref(0, shape_input)
    b.add_field_int8(1, 0)      # FLOAT32
    b.add_field_uint32(2, 2)    # buffer 2
    b.add_field_ref(3, str_input)
    tensor0 = b.end_object()

    shape_weight = b.create_vector('<i', list(shape))
    b.start_object(10)
    b.add_field_ref(0, shape_weight)
    b.add_field_int8(1, 0)      # FLOAT32
    b.add_field_uint32(2, 1)    # buffer 1 (sparse values)
    b.add_field_ref(3, str_weight)
    b.add_field_ref(6, sparsity_params)
    tensor1 = b.end_object()

    shape_output = b.create_vector('<i', [1, shape[1]])
    b.start_object(10)
    b.add_field_ref(0, shape_output)
    b.add_field_int8(1, 0)      # FLOAT32
    b.add_field_uint32(2, 3)    # buffer 3
    b.add_field_ref(3, str_output)
    tensor2 = b.end_object()

    # OperatorCode: ADD
    b.start_object(4)
    b.add_field_int8(0, 0)
    b.add_field_int32(2, 1)
    b.add_field_int32(3, 0)
    opcode = b.end_object()

    # Operator
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
    desc = b.create_string("sparse_test")

    b.start_object(5)
    b.add_field_uint32(0, 3)
    b.add_field_ref(1, opcodes_vec)
    b.add_field_ref(2, subgraphs_vec)
    b.add_field_ref(3, desc)
    b.add_field_ref(4, buffers_vec)
    model = b.end_object()

    b.finish(model, "TFL3")
    return b.output()


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        sys.exit(1)

    path_to_model_dir = os.path.join(sys.argv[1], "sparse_oob")
    os.makedirs(path_to_model_dir, exist_ok=True)

    # 1. OOB index value: index 999 in dimension of size 100 (valid range 0-99)
    #    Without fix this causes heap OOB write: row_offset = 999 * 4 = 3996, buffer = 800 bytes
    data = build_sparse_model(
        indices=[5, 999, 1],       # 999 is out of bounds
        segments=[0, 2, 3],        # row 0: 2 elements, row 1: 1 element
        shape=(2, 100),
        num_values=3
    )
    with open(os.path.join(path_to_model_dir, 'sparse_oob_index.tflite'), 'wb') as f:
        f.write(data)

    # 2. Negative index value: index -1 (invalid for unsigned column offset)
    data = build_sparse_model(
        indices=[5, -1, 1],        # -1 is a negative index
        segments=[0, 2, 3],
        shape=(2, 100),
        num_values=3
    )
    with open(os.path.join(path_to_model_dir, 'sparse_negative_index.tflite'), 'wb') as f:
        f.write(data)

    # 3. Non-monotonic segments: segments go 0, 3, 2 (decreasing)
    data = build_sparse_model(
        indices=[5, 1, 2],
        segments=[0, 3, 2],        # non-monotonic: 3 > 2
        shape=(2, 100),
        num_values=3
    )
    with open(os.path.join(path_to_model_dir, 'sparse_non_monotonic_segments.tflite'), 'wb') as f:
        f.write(data)

    # 4. Overflow shape: dimensions that cause size_t overflow in total_size computation
    #    Shape [2147483647, 2147483647] with FLOAT32 (4 bytes) -> total_size overflows
    data = build_sparse_model(
        indices=[1],
        segments=[0, 1],
        shape=(2147483647, 2147483647),
        num_values=1
    )
    with open(os.path.join(path_to_model_dir, 'sparse_overflow_shape.tflite'), 'wb') as f:
        f.write(data)


if __name__ == "__main__":
    main()
