import onnxruntime as ort
import openvino as ov
import numpy as np
import onnx
from onnx import helper, TensorProto

def patch_model(model_path, output_path):
    model = onnx.load(model_path)
    g = model.graph

    # find ReverseSequence
    rev_node = next(n for n in g.node if n.op_type == "ReverseSequence")
    x_name, seq_name = rev_node.input[0], rev_node.input[1]
    out_name = rev_node.output[0]
    ba = next(attr.i for attr in rev_node.attribute if attr.name == "batch_axis")
    ta = next(attr.i for attr in rev_node.attribute if attr.name == "time_axis")

    # input rank & shape info
    x_shape_info = next(i for i in g.input if i.name == x_name).type.tensor_type.shape.dim
    rank = len(x_shape_info)

    # zero scalar node
    zero_node = helper.make_node("Constant", [], [f"{out_name}_zero"],
                                 value=helper.make_tensor("z", TensorProto.FLOAT, [], [0.0]))

    # cast seq -> float
    cast_node = helper.make_node("Cast", [seq_name], [f"{seq_name}_f"], to=TensorProto.FLOAT)

    # equal to zero -> produces [B]
    eq_node = helper.make_node("Equal", [cast_node.output[0], zero_node.output[0]], [f"{seq_name}_is_zero"])

    # --- Unsqueeze mask to shape placing batch axis at correct position ---
    # We need to insert axes at every dimension index except batch_axis.
    mask_input = eq_node.output[0]
    mask_nodes = []
    # build a list of axes to unsqueeze: all indices 0..rank-1 except batch axis
    unsq_axes = [i for i in range(rank) if i != ba]
    # unsqueeze in increasing order so inserted axes indices remain correct
    for idx, axis in enumerate(unsq_axes):
        axes_const_name = f"mask_axes_{idx}"
        axes_const = helper.make_node("Constant", [], [axes_const_name],
                                      value=helper.make_tensor("a", TensorProto.INT64, [1], [axis]))
        unsq_name = f"{seq_name}_mask_unsq_{idx}"
        unsq = helper.make_node("Unsqueeze", [mask_input, axes_const.output[0]], [unsq_name])
        mask_input = unsq.output[0]
        mask_nodes.extend([axes_const, unsq])
    mask_expanded = mask_input  # now shape matches x except dims at batch_axis are B and others 1

    # --- Unsqueeze zero scalar to [1,1,...] using all axes 0..rank-1 ---
    zero_input = zero_node.output[0]
    zero_nodes = []
    for idx, axis in enumerate(range(rank)):
        axes_const_name = f"zero_axes_{idx}"
        axes_const = helper.make_node("Constant", [], [axes_const_name],
                                      value=helper.make_tensor("a", TensorProto.INT64, [1], [axis]))
        unsq_name = f"{out_name}_zero_unsq_{idx}"
        unsq = helper.make_node("Unsqueeze", [zero_input, axes_const.output[0]], [unsq_name])
        zero_input = unsq.output[0]
        zero_nodes.extend([axes_const, unsq])
    zero_expanded = zero_input  # shape [1,1,...,1]

    # dynamic input shape: Shape(x)
    shape_of = helper.make_node("Shape", [x_name], [f"{x_name}_shape"])

    # Expand mask and zero to full shape using dynamic Shape(x)
    expand_mask = helper.make_node("Expand", [mask_expanded, shape_of.output[0]], [f"{seq_name}_mask_full"])
    expand_zero = helper.make_node("Expand", [zero_expanded, shape_of.output[0]], [f"{out_name}_zero_full"])

    # make ReverseSequence write to temp
    rev_temp = f"{out_name}_temp"
    rev_node.output[0] = rev_temp

    # Where(mask, zero, rev_temp) -> final output
    where_node = helper.make_node("Where", [expand_mask.output[0], expand_zero.output[0], rev_temp], [f"{out_name}_final"])

    # assemble nodes (preserve order: helpers -> unsqueezes -> shape -> expand -> rev -> where)
    g.node.remove(rev_node)
    g.node.extend([zero_node, cast_node, eq_node] + mask_nodes + zero_nodes + [shape_of, expand_mask, expand_zero, rev_node, where_node])

    # fix graph output reference
    for o in g.output:
        if o.name == out_name:
            o.name = f"{out_name}_final"

    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


# === TEST CASES ===
cases = [
    (1, 0, [4, 4], [2, 0, 1, 0]),    # case 1: previously passed
    (0, 1, [4, 4], [2, 0, 1, 0]),    # case 2: previously passed
    (1, 0, [2, 3, 4], [1, 0, 2]),    # case 3: 3D
]

for i, (ba, ta, shape, seq_vals) in enumerate(cases):
    print(f"\n=== CASE {i+1}: batch_axis={ba}, time_axis={ta}, shape={shape} ===")
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    seq = np.array(seq_vals, np.int64)

    # build simple ONNX model with ReverseSequence
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    seq_info = helper.make_tensor_value_info("seq", TensorProto.INT64, [len(seq_vals)])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)
    node = helper.make_node("ReverseSequence", ["x", "seq"], ["y"], batch_axis=ba, time_axis=ta)
    graph = helper.make_graph([node], "test", [x_info, seq_info], [y_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, "temp.onnx")

    try:
        patch_model("temp.onnx", "temp_fixed.onnx")
        ort_out = ort.InferenceSession("temp.onnx").run(None, {"x": x, "seq": seq})[0]
        ov_model = ov.convert_model("temp_fixed.onnx")
        compiled = ov.Core().compile_model(ov_model, "CPU")
        ov_out = compiled({"x": x, "seq": seq})[0]
        print("ORT:\n", np.round(ort_out, 1))
        print("OV:\n", np.round(ov_out, 1))
        np.testing.assert_allclose(ort_out, ov_out, atol=1e-6)
        print("✅ PASS")
    except Exception as e:
        print("❌ FAIL:", str(e)[:500])
