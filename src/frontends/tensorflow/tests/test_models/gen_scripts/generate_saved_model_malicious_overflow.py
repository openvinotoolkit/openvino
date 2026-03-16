# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import struct
import sys


def encode_varint(value):
    """Encode an integer as a protobuf varint."""
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def encode_signed_varint(value):
    """Encode a signed int64 as a protobuf varint (two's complement)."""
    if value < 0:
        value = (1 << 64) + value
    return encode_varint(value)


def encode_field(field_number, wire_type, data):
    """Encode a protobuf field."""
    tag = (field_number << 3) | wire_type
    return encode_varint(tag) + data


def encode_varint_field(field_number, value):
    return encode_field(field_number, 0, encode_varint(value))


def encode_signed_varint_field(field_number, value):
    return encode_field(field_number, 0, encode_signed_varint(value))


def encode_bytes_field(field_number, data):
    return encode_field(field_number, 2, encode_varint(len(data)) + data)


def encode_string_field(field_number, s):
    return encode_bytes_field(field_number, s.encode('utf-8'))


def encode_fixed32_field(field_number, value):
    return encode_field(field_number, 5, struct.pack('<I', value))


def build_bundle_header_proto():
    """BundleHeaderProto: num_shards=1, endianness=LITTLE, version={producer=1, min_consumer=0}"""
    version = encode_varint_field(1, 1) + encode_varint_field(2, 0)
    return encode_varint_field(1, 1) + encode_varint_field(2, 0) + encode_bytes_field(3, version)


def build_bundle_entry_proto(dtype, shape_dims, shard_id, offset, size):
    """BundleEntryProto with given parameters."""
    shape_bytes = b''
    for d in shape_dims:
        dim_msg = encode_signed_varint_field(1, d)
        shape_bytes += encode_bytes_field(2, dim_msg)
    return (encode_varint_field(1, dtype) +
            encode_bytes_field(2, shape_bytes) +
            encode_varint_field(3, shard_id) +
            encode_signed_varint_field(4, offset) +
            encode_signed_varint_field(5, size) +
            encode_fixed32_field(6, 0))


def build_sstable_block(entries):
    """Build an SSTable data block from key-value pairs."""
    block = bytearray()
    restart_offsets = []
    prev_key = b''
    for i, (key, value) in enumerate(entries):
        key_bytes = key.encode('utf-8') if isinstance(key, str) else key
        shared = 0
        for j in range(min(len(prev_key), len(key_bytes))):
            if prev_key[j] == key_bytes[j]:
                shared += 1
            else:
                break
        non_shared = len(key_bytes) - shared
        if i % 16 == 0:
            restart_offsets.append(len(block))
            shared = 0
            non_shared = len(key_bytes)
        block.extend(encode_varint(shared))
        block.extend(encode_varint(non_shared))
        block.extend(encode_varint(len(value)))
        block.extend(key_bytes[shared:])
        block.extend(value)
        prev_key = key_bytes
    if not restart_offsets:
        restart_offsets = [0]
    for off in restart_offsets:
        block.extend(struct.pack('<I', off))
    block.extend(struct.pack('<I', len(restart_offsets)))
    return bytes(block)


def build_sstable_footer(index_block_offset, index_block_size,
                         metaindex_block_offset=0, metaindex_block_size=0):
    """Build SSTable footer (48 bytes)."""
    metaindex_handle = encode_varint(metaindex_block_offset) + encode_varint(metaindex_block_size)
    index_handle = encode_varint(index_block_offset) + encode_varint(index_block_size)
    footer = bytearray()
    footer.extend(metaindex_handle)
    footer.extend(index_handle)
    while len(footer) < 40:
        footer.append(0)
    footer.extend(struct.pack('<Q', 0xdb4775248b80fb57))
    return bytes(footer)


def build_saved_model_pb():
    """Build minimal saved_model.pb with VarHandleOp using raw protobuf encoding."""
    DT_FLOAT = 1
    dtype_attr_value = encode_varint_field(6, DT_FLOAT)
    dim2 = encode_signed_varint_field(1, 2)
    shape_proto = encode_bytes_field(2, dim2) + encode_bytes_field(2, dim2)
    shape_attr_value = encode_bytes_field(7, shape_proto)
    t_attr_value = encode_varint_field(6, DT_FLOAT)

    def build_node_def(name, op, inputs=None, attrs=None):
        msg = encode_string_field(1, name) + encode_string_field(2, op)
        if inputs:
            for inp in inputs:
                msg += encode_string_field(3, inp)
        if attrs:
            for key, val in attrs.items():
                map_entry = encode_string_field(1, key) + encode_bytes_field(2, val)
                msg += encode_bytes_field(5, map_entry)
        return msg

    varhandle_node = build_node_def(
        "my_var", "VarHandleOp",
        attrs={"dtype": dtype_attr_value, "shape": shape_attr_value})
    readvariable_node = build_node_def(
        "read_var", "ReadVariableOp",
        inputs=["my_var"],
        attrs={"dtype": dtype_attr_value})
    identity_node = build_node_def(
        "Identity", "Identity",
        inputs=["read_var"],
        attrs={"T": t_attr_value})

    graph_def = (encode_bytes_field(1, varhandle_node) +
                 encode_bytes_field(1, readvariable_node) +
                 encode_bytes_field(1, identity_node))
    meta_info_def = encode_string_field(1, "serve")
    tensor_info = encode_string_field(1, "Identity:0") + encode_varint_field(2, DT_FLOAT)
    sig_output_entry = encode_string_field(1, "output_0") + encode_bytes_field(2, tensor_info)
    signature_def = encode_bytes_field(2, sig_output_entry)
    sig_map_entry = encode_string_field(1, "serving_default") + encode_bytes_field(2, signature_def)
    meta_graph_def = (encode_bytes_field(1, meta_info_def) +
                      encode_bytes_field(2, graph_def) +
                      encode_bytes_field(5, sig_map_entry))
    return encode_signed_varint_field(1, 1) + encode_bytes_field(2, meta_graph_def)


def main():
    output_dir = os.path.join(sys.argv[1], "saved_model_malicious_overflow")
    os.makedirs(os.path.join(output_dir, "variables"), exist_ok=True)

    # Malicious offset that causes signed int64 overflow when added to size
    malicious_offset = 0x7FFFFFFFFFFFFFF0

    # Build variables.index SSTable with 3 entries:
    # "" -> BundleHeaderProto
    # "_CHECKPOINTABLE_OBJECT_GRAPH" -> BundleEntryProto (malicious offset)
    # "my_var" -> BundleEntryProto (malicious offset)
    header_proto = build_bundle_header_proto()
    ckog_entry = build_bundle_entry_proto(
        dtype=7, shape_dims=[], shard_id=0,
        offset=malicious_offset, size=16)
    var_entry = build_bundle_entry_proto(
        dtype=1, shape_dims=[2, 2], shard_id=0,
        offset=malicious_offset, size=16)

    data_entries = [
        ("", header_proto),
        ("_CHECKPOINTABLE_OBJECT_GRAPH", ckog_entry),
        ("my_var", var_entry),
    ]
    data_block = build_sstable_block(data_entries)
    data_block_full = data_block + b'\x00' + struct.pack('<I', 0)

    metaindex_block = build_sstable_block([])
    metaindex_full = metaindex_block + b'\x00' + struct.pack('<I', 0)

    last_key = "my_var"
    data_block_handle = encode_varint(0) + encode_varint(len(data_block))
    index_block = build_sstable_block([(last_key, data_block_handle)])
    index_full = index_block + b'\x00' + struct.pack('<I', 0)

    metaindex_offset = len(data_block_full)
    index_offset = metaindex_offset + len(metaindex_full)

    footer = build_sstable_footer(
        index_block_offset=index_offset,
        index_block_size=len(index_block),
        metaindex_block_offset=metaindex_offset,
        metaindex_block_size=len(metaindex_block))

    index_file = data_block_full + metaindex_full + index_full + footer

    with open(os.path.join(output_dir, "saved_model.pb"), 'wb') as f:
        f.write(build_saved_model_pb())
    with open(os.path.join(output_dir, "variables", "variables.index"), 'wb') as f:
        f.write(index_file)
    with open(os.path.join(output_dir, "variables", "variables.data-00000-of-00001"), 'wb') as f:
        f.write(b'\x00' * 64)


if __name__ == '__main__':
    main()
