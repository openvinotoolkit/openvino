// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/tile.hpp"
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_REPEAT tiles src[0] to fill the destination shape; every destination dimension is an
// integer multiple of the corresponding source dimension.
OutputVector translate_repeat(const NodeContext& context) {
    num_inputs_check(context, 1, 2);

    auto input = context.get_input(0);
    const auto input_shape = context.get_input_shape(0);
    const auto output_shape = context.get_output_shape();

    // Preferred path: build static integer repeat factors and let OV shape-inference propagate any
    // dynamic axis. REPEAT never tiles the variable token/sequence axis, so a dynamic dim (present on
    // both input and output at the same position) always has repeat factor 1; only static dims are
    // tiled. This keeps the Tile's repeat vector constant even on the dynamic-token path (the fused
    // qwen3-next conv/delta REPEATs have a dynamic token axis and only one input, so a ShapeOf-based
    // fallback that reads a non-existent src[1] would produce garbage).
    if (input_shape.rank().is_static() && output_shape.rank().is_static() &&
        input_shape.rank() == output_shape.rank()) {
        const auto rank = static_cast<size_t>(input_shape.rank().get_length());
        std::vector<int64_t> repeats(rank, 1);
        bool resolved = true;

        for (size_t axis = 0; axis < rank; ++axis) {
            const bool in_dyn = !input_shape[axis].is_static();
            const bool out_dyn = !output_shape[axis].is_static();
            if (in_dyn || out_dyn) {
                // A dynamic axis is the token axis, which REPEAT passes through untiled.
                repeats[axis] = 1;
                continue;
            }
            const int64_t input_dim = input_shape[axis].get_length();
            const int64_t output_dim = output_shape[axis].get_length();
            if (input_dim <= 0 || output_dim <= 0 || output_dim % input_dim != 0) {
                resolved = false;
                break;
            }
            repeats[axis] = output_dim / input_dim;
        }

        if (resolved) {
            auto repeats_node = ov::op::v0::Constant::create(ov::element::i64, {repeats.size()}, repeats);
            ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(input, repeats_node);
            return rename_outputs_with_suffix({res}, context.get_name());
        }
    }

    // Dynamic-rank fallback: tile by the ratio of the (statically known) output shape to the runtime
    // input shape. Only reached when the output shape is fully static but ranks/dims did not resolve
    // above; a two-input REPEAT could instead take its target from src[1].
    auto input_shape_node = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    std::shared_ptr<ov::Node> target_shape_node;
    if (output_shape.is_static()) {
        target_shape_node =
            ov::op::v0::Constant::create(ov::element::i64, {output_shape.to_shape().size()}, output_shape.to_shape());
    } else {
        FRONT_END_OP_CONVERSION_CHECK(context.get_input_size() > 1,
                                      "REPEAT with dynamic output shape requires a second (shape) input");
        target_shape_node = std::make_shared<ov::op::v3::ShapeOf>(context.get_input(1), ov::element::i64);
    }
    auto repeats_node = std::make_shared<ov::op::v1::Divide>(target_shape_node, input_shape_node);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v0::Tile>(input, repeats_node);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
