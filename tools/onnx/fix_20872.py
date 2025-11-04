#!/usr/bin/env python3
"""
Patch ONNX ReverseSequence to zero output when sequence_lens[i] == 0
Fixes OpenVINO vs ONNX Runtime mismatch (#20872)
Works for any batch_axis, time_axis, rank, and dynamic shapes.
"""

import onnx
from onnx import helper, TensorProto

def patch_reverse_sequence_zero_len(model_path: str, output_path: str):
    model = onnx.load(model_path)
    g = model.graph

    # Find ReverseSequence
    rev_node = next(n for n in g.node if n.op_type == "ReverseSequence")
    x_name, seq_name = rev_node.input[0], rev_node.input[1]
    out_name = rev_node.output[0]
    ba = next(attr.i for attr in rev_node.attribute if attr.name == "batch_axis")
    ta = next(attr.i for attr in rev_node.attribute if attr.name == "time_axis")

    # Input rank
    x_shape_info = next(i for i in g.input if i.name == x_name).type.tensor_type.shape.dim
    rank = len(x_shape_info)

    # === Nodes ===
    zero_node = helper.make_node("Constant", [], [f"{out_name}_zero"],
                                value=helper.make_tensor("z", TensorProto.FLOAT, [], [0.0]))
    cast_node = helper.make_node("Cast", [seq_name], [f"{seq_name}_f"], to=TensorProto.FLOAT)
    eq_node = helper.make_node("Equal", [cast_node.output[0], zero_node.output[0]], [f"{seq_name}_is_zero"])

    # Unsqueeze mask: insert 1s in all dims except batch_axis
    mask_input = eq_node.output[0]
    mask_nodes = []
    unsq_axes = [i for i in range(rank) if i != ba]
    for idx, axis in enumerate(unsq_axes):
        axes_const = helper.make_node("Constant", [], [f"mask_axes_{idx}"],
                                     value=helper.make_tensor("a", TensorProto.INT64, [1], [axis]))
        unsq = helper.make_node("Unsqueeze", [mask_input, axes_const.output[0]], [f"{seq_name}_mask_unsq_{idx}"])
        mask_input = unsq.output[0]
        mask_nodes.extend([axes_const, unsq])
    mask_expanded = mask_input

    # Unsqueeze zero scalar to [1,1,...,1]
    zero_input = zero_node.output[0]
    zero_nodes = []
    for idx in range(rank):
        axes_const = helper.make_node("Constant", [], [f"zero_axes_{idx}"],
                                     value=helper.make_tensor("a", TensorProto.INT64, [1], [idx]))
        unsq = helper.make_node("Unsqueeze", [zero_input, axes_const.output[0]], [f"{out_name}_zero_unsq_{idx}"])
        zero_input = unsq.output[0]
        zero_nodes.extend([axes_const, unsq])
    zero_expanded = zero_input

    # Dynamic shape
    shape_of = helper.make_node("Shape", [x_name], [f"{x_name}_shape"])

    # Expand
    expand_mask = helper.make_node("Expand", [mask_expanded, shape_of.output[0]], [f"{seq_name}_mask_full"])
    expand_zero = helper.make_node("Expand", [zero_expanded, shape_of.output[0]], [f"{out_name}_zero_full"])

    # Temp output
    rev_temp = f"{out_name}_temp"
    rev_node.output[0] = rev_temp

    # Where
    where_node = helper.make_node("Where", [expand_mask.output[0], expand_zero.output[0], rev_temp], [f"{out_name}_final"])

    # Rebuild graph
    g.node.remove(rev_node)
    g.node.extend([zero_node, cast_node, eq_node] + mask_nodes + zero_nodes +
                  [shape_of, expand_mask, expand_zero, rev_node, where_node])

    # Fix output
    for o in g.output:
        if o.name == out_name:
            o.name = f"{out_name}_final"

    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Patched model saved: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 fix_20872.py <input.onnx> <output.onnx>")
        sys.exit(1)
    patch_reverse_sequence_zero_len(sys.argv[1], sys.argv[2])
