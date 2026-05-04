# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# What: Generates 4 malicious TF SavedModel directories that trigger the OOB read in
#       VariablesIndex::map_assignvariable() (variables_index.cpp).
#       No TF runtime needed — all protobuf and SSTable structures are built by hand.
#
# How:  Each SavedModel has:
#         saved_model.pb              — GraphDef with nodes described per variant below.
#         variables/variables.index   — Minimal SSTable (bundle header only, num_shards=1).
#         variables/variables.data-*  — 64 zero bytes (needed to pass open-file checks).
#
#       Four variants:
#         saved_model_oob_pos_index/    — TF2 (AssignVariableOp path):
#                                         Identity.input = "save/RestoreV2:999",
#                                         tensor_names has 1 entry → positive OOB index
#         saved_model_oob_neg_index/    — TF2 (AssignVariableOp path):
#                                         Identity.input = "save/RestoreV2:-1",
#                                         tensor_names has 1 entry → negative OOB index
#         saved_model_oob_empty_names/  — TF2 (AssignVariableOp path):
#                                         Identity.input = "save/RestoreV2:0",
#                                         tensor_names has 0 entries → OOB via implicit-0
#                                         path when string_val_size()==0
#         saved_model_oob_assign_path/  — TF1 (Assign path):
#                                         Assign.input(1) = "save/RestoreV2:999",
#                                         tensor_names has 1 entry → positive OOB index
#
# Why:  SSTable builder is copied verbatim from generate_saved_model_malicious_overflow.py.
#       Generator is run by the CMake build system.
#
# Usage: python3 generate_saved_model_oob_index.py <output_dir>
#        Output dirs are created under <output_dir>/.

import os
import struct
import sys


# ---------------------------------------------------------------------------
# Protobuf varint / field encoding (copied from generate_saved_model_malicious_overflow.py)
# ---------------------------------------------------------------------------

def encode_varint(value):
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)


def encode_signed_varint(value):
    if value < 0:
        value = (1 << 64) + value
    return encode_varint(value)


def encode_field(field_number, wire_type, data):
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


# ---------------------------------------------------------------------------
# SSTable builder (copied from generate_saved_model_malicious_overflow.py)
# ---------------------------------------------------------------------------

def build_sstable_block(entries):
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
    metaindex_handle = encode_varint(metaindex_block_offset) + encode_varint(metaindex_block_size)
    index_handle = encode_varint(index_block_offset) + encode_varint(index_block_size)
    footer = bytearray()
    footer.extend(metaindex_handle)
    footer.extend(index_handle)
    while len(footer) < 40:
        footer.append(0)
    footer.extend(struct.pack('<Q', 0xdb4775248b80fb57))
    return bytes(footer)


def build_bundle_header_proto():
    """BundleHeaderProto: num_shards=1, endianness=LITTLE, version={producer=1,min_consumer=0}"""
    version = encode_varint_field(1, 1) + encode_varint_field(2, 0)
    return encode_varint_field(1, 1) + encode_varint_field(2, 0) + encode_bytes_field(3, version)


def build_minimal_variables_index():
    """Minimal SSTable with only the bundle header key. num_shards=1."""
    header_proto = build_bundle_header_proto()
    data_entries = [("", header_proto)]
    data_block = build_sstable_block(data_entries)
    data_block_full = data_block + b'\x00' + struct.pack('<I', 0)

    metaindex_block = build_sstable_block([])
    metaindex_full = metaindex_block + b'\x00' + struct.pack('<I', 0)

    last_key = ""
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

    return data_block_full + metaindex_full + index_full + footer


# ---------------------------------------------------------------------------
# GraphDef / SavedModel builder
# ---------------------------------------------------------------------------

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


def build_saved_model_pb(identity_input, num_string_vals=1):
    """
    Build saved_model.pb with:
      VarHandleOp("my_var")
      Const("tensor_names")  — num_string_vals string_val entries
      RestoreV2("save/RestoreV2")
      Identity("save/identity")  — input[0] = identity_input
      AssignVariableOp("save/Assign")

    identity_input controls the RestoreV2 output index extracted by parse_node_name().
    num_string_vals controls how many entries tensor_names has (0 = empty → OOB at index 0).
    """
    DT_FLOAT = 1
    DT_STRING = 7

    # VarHandleOp attrs
    dtype_attr = encode_varint_field(6, DT_FLOAT)
    dim2 = encode_signed_varint_field(1, 2)
    shape_proto = encode_bytes_field(2, dim2) + encode_bytes_field(2, dim2)
    shape_attr = encode_bytes_field(7, shape_proto)

    varhandle_node = build_node_def(
        "my_var", "VarHandleOp",
        attrs={"dtype": dtype_attr, "shape": shape_attr,
               "container": encode_string_field(2, ""),
               "shared_name": encode_string_field(2, "my_var")})

    # Const("tensor_names") — TensorProto with num_string_vals string_val entries
    actual_size = max(num_string_vals, 0)
    tensor_shape = encode_bytes_field(2, encode_signed_varint_field(1, actual_size))
    tensor_proto = encode_varint_field(1, DT_STRING) + encode_bytes_field(4, tensor_shape)
    for _ in range(actual_size):
        tensor_proto += encode_bytes_field(8, b"Variable")
    tensor_attr = encode_bytes_field(8, tensor_proto)
    dtype_str_attr = encode_varint_field(6, DT_STRING)

    const_node = build_node_def(
        "tensor_names", "Const",
        attrs={"value": tensor_attr, "dtype": dtype_str_attr})

    # Const("shape_and_slices")
    shape_tensor_shape = encode_bytes_field(2, encode_signed_varint_field(1, actual_size))
    empty_shape_tensor = encode_varint_field(1, DT_STRING) + encode_bytes_field(4, shape_tensor_shape)
    for _ in range(actual_size):
        empty_shape_tensor += encode_bytes_field(8, b"")
    empty_shape_attr = encode_bytes_field(8, empty_shape_tensor)

    shape_slices_node = build_node_def(
        "shape_and_slices", "Const",
        attrs={"value": empty_shape_attr, "dtype": dtype_str_attr})

    # Const("prefix")
    scalar_shape = encode_bytes_field(4, b"")
    prefix_tensor = (encode_varint_field(1, DT_STRING) +
                     scalar_shape +
                     encode_bytes_field(8, b"checkpoint"))
    prefix_attr = encode_bytes_field(8, prefix_tensor)

    prefix_node = build_node_def(
        "prefix", "Const",
        attrs={"value": prefix_attr, "dtype": dtype_str_attr})

    # RestoreV2
    list_attr = encode_bytes_field(5, encode_varint_field(6, DT_FLOAT))
    restorev2_node = build_node_def(
        "save/RestoreV2", "RestoreV2",
        inputs=["prefix", "tensor_names", "shape_and_slices"],
        attrs={"dtypes": list_attr})

    # Identity
    t_attr = encode_varint_field(6, DT_FLOAT)
    identity_node = build_node_def(
        "save/identity", "Identity",
        inputs=[identity_input],
        attrs={"T": t_attr})

    # AssignVariableOp
    assign_node = build_node_def(
        "save/Assign", "AssignVariableOp",
        inputs=["my_var", "save/identity"],
        attrs={"dtype": dtype_attr})

    # ReadVariableOp + Identity for model output
    read_node = build_node_def(
        "read_var", "ReadVariableOp",
        inputs=["my_var"],
        attrs={"dtype": dtype_attr})

    output_identity = build_node_def(
        "Identity", "Identity",
        inputs=["read_var"],
        attrs={"T": t_attr})

    graph_def = (encode_bytes_field(1, varhandle_node) +
                 encode_bytes_field(1, const_node) +
                 encode_bytes_field(1, shape_slices_node) +
                 encode_bytes_field(1, prefix_node) +
                 encode_bytes_field(1, restorev2_node) +
                 encode_bytes_field(1, identity_node) +
                 encode_bytes_field(1, assign_node) +
                 encode_bytes_field(1, read_node) +
                 encode_bytes_field(1, output_identity))

    meta_info_def = encode_string_field(1, "serve")
    tensor_info = encode_string_field(1, "Identity:0") + encode_varint_field(2, DT_FLOAT)
    sig_output_entry = encode_string_field(1, "output_0") + encode_bytes_field(2, tensor_info)
    signature_def = encode_bytes_field(2, sig_output_entry)
    sig_map_entry = encode_string_field(1, "serving_default") + encode_bytes_field(2, signature_def)
    meta_graph_def = (encode_bytes_field(1, meta_info_def) +
                      encode_bytes_field(2, graph_def) +
                      encode_bytes_field(5, sig_map_entry))

    return encode_signed_varint_field(1, 1) + encode_bytes_field(2, meta_graph_def)


def build_saved_model_pb_tf1_assign(index_str, num_string_vals=1):
    """
    Build saved_model.pb with TF1-style topology:
      VariableV2("my_variable")
      Const("tensor_names")  — num_string_vals string_val entries
      RestoreV2("save/RestoreV2")
      Assign("save/Assign")  — input(1) = index_str (e.g. "save/RestoreV2:999")
      Identity("Identity")   — output node

    This exercises the Assign code path in VariablesIndex::map_assignvariable()
    rather than the AssignVariableOp path exercised by build_saved_model_pb().
    """
    DT_FLOAT = 1
    DT_STRING = 7

    # VariableV2 attrs
    dtype_attr = encode_varint_field(6, DT_FLOAT)
    dim_attr = encode_bytes_field(2, encode_signed_varint_field(1, 4))
    shape_attr = encode_bytes_field(7, dim_attr)

    variablev2_node = build_node_def(
        "my_variable", "VariableV2",
        attrs={"dtype": dtype_attr, "shape": shape_attr,
               "container": encode_string_field(2, ""),
               "shared_name": encode_string_field(2, "")})

    # Const("tensor_names")
    actual_size = max(num_string_vals, 0)
    tensor_shape = encode_bytes_field(2, encode_signed_varint_field(1, actual_size))
    tensor_proto = encode_varint_field(1, DT_STRING) + encode_bytes_field(4, tensor_shape)
    for _ in range(actual_size):
        tensor_proto += encode_bytes_field(8, b"Variable")
    tensor_attr = encode_bytes_field(8, tensor_proto)
    dtype_str_attr = encode_varint_field(6, DT_STRING)

    const_node = build_node_def(
        "tensor_names", "Const",
        attrs={"value": tensor_attr, "dtype": dtype_str_attr})

    # Const("shape_and_slices")
    shape_tensor_shape = encode_bytes_field(2, encode_signed_varint_field(1, actual_size))
    empty_shape_tensor = encode_varint_field(1, DT_STRING) + encode_bytes_field(4, shape_tensor_shape)
    for _ in range(actual_size):
        empty_shape_tensor += encode_bytes_field(8, b"")
    empty_shape_attr = encode_bytes_field(8, empty_shape_tensor)

    shape_slices_node = build_node_def(
        "shape_and_slices", "Const",
        attrs={"value": empty_shape_attr, "dtype": dtype_str_attr})

    # Const("prefix")
    scalar_shape = encode_bytes_field(4, b"")
    prefix_tensor = (encode_varint_field(1, DT_STRING) +
                     scalar_shape +
                     encode_bytes_field(8, b"checkpoint"))
    prefix_attr = encode_bytes_field(8, prefix_tensor)

    prefix_node = build_node_def(
        "prefix", "Const",
        attrs={"value": prefix_attr, "dtype": dtype_str_attr})

    # RestoreV2
    list_attr = encode_bytes_field(5, encode_varint_field(6, DT_FLOAT))
    restorev2_node = build_node_def(
        "save/RestoreV2", "RestoreV2",
        inputs=["prefix", "tensor_names", "shape_and_slices"],
        attrs={"dtypes": list_attr})

    # Assign: input(0) = variable ref, input(1) = RestoreV2 output with OOB index
    t_attr = encode_varint_field(6, DT_FLOAT)
    assign_node = build_node_def(
        "save/Assign", "Assign",
        inputs=["my_variable", index_str],
        attrs={"T": t_attr})

    # Identity for signature output
    output_identity = build_node_def(
        "Identity", "Identity",
        inputs=["my_variable"],
        attrs={"T": t_attr})

    graph_def = (encode_bytes_field(1, variablev2_node) +
                 encode_bytes_field(1, const_node) +
                 encode_bytes_field(1, shape_slices_node) +
                 encode_bytes_field(1, prefix_node) +
                 encode_bytes_field(1, restorev2_node) +
                 encode_bytes_field(1, assign_node) +
                 encode_bytes_field(1, output_identity))

    meta_info_def = encode_string_field(1, "serve")
    tensor_info = encode_string_field(1, "Identity:0") + encode_varint_field(2, DT_FLOAT)
    sig_output_entry = encode_string_field(1, "output_0") + encode_bytes_field(2, tensor_info)
    signature_def = encode_bytes_field(2, sig_output_entry)
    sig_map_entry = encode_string_field(1, "serving_default") + encode_bytes_field(2, signature_def)
    meta_graph_def = (encode_bytes_field(1, meta_info_def) +
                      encode_bytes_field(2, graph_def) +
                      encode_bytes_field(5, sig_map_entry))

    return encode_signed_varint_field(1, 1) + encode_bytes_field(2, meta_graph_def)


def write_savedmodel(output_dir, identity_input, num_string_vals=1):
    os.makedirs(os.path.join(output_dir, "variables"), exist_ok=True)
    with open(os.path.join(output_dir, "saved_model.pb"), 'wb') as f:
        f.write(build_saved_model_pb(identity_input, num_string_vals))
    with open(os.path.join(output_dir, "variables", "variables.index"), 'wb') as f:
        f.write(build_minimal_variables_index())
    with open(os.path.join(output_dir, "variables", "variables.data-00000-of-00001"), 'wb') as f:
        f.write(b'\x00' * 64)


def write_savedmodel_tf1_assign(output_dir, index_str, num_string_vals=1):
    os.makedirs(os.path.join(output_dir, "variables"), exist_ok=True)
    with open(os.path.join(output_dir, "saved_model.pb"), 'wb') as f:
        f.write(build_saved_model_pb_tf1_assign(index_str, num_string_vals))
    with open(os.path.join(output_dir, "variables", "variables.index"), 'wb') as f:
        f.write(build_minimal_variables_index())
    with open(os.path.join(output_dir, "variables", "variables.data-00000-of-00001"), 'wb') as f:
        f.write(b'\x00' * 64)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir>")
        sys.exit(1)

    base = sys.argv[1]

    # Positive OOB: index 999 into a tensor with 1 string_val
    write_savedmodel(os.path.join(base, "saved_model_oob_pos_index"), "save/RestoreV2:999")
    print(f"Generated: {os.path.join(base, 'saved_model_oob_pos_index')}")

    # Negative OOB: index -1 into a tensor with 1 string_val
    write_savedmodel(os.path.join(base, "saved_model_oob_neg_index"), "save/RestoreV2:-1")
    print(f"Generated: {os.path.join(base, 'saved_model_oob_neg_index')}")

    # Empty tensor_names: index 0 into a tensor with 0 string_val entries;
    # exercises the upper-bound guard on the implicit-0 code path
    write_savedmodel(os.path.join(base, "saved_model_oob_empty_names"), "save/RestoreV2:0",
                     num_string_vals=0)
    print(f"Generated: {os.path.join(base, 'saved_model_oob_empty_names')}")

    # TF1-style Assign path: OOB positive index in the Assign branch of map_assignvariable()
    write_savedmodel_tf1_assign(os.path.join(base, "saved_model_oob_assign_path"),
                                "save/RestoreV2:999")
    print(f"Generated: {os.path.join(base, 'saved_model_oob_assign_path')}")


if __name__ == '__main__':
    main()
