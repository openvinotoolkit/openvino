// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_RESHAPE. The decoder describes the target layout; nothing is inferred from a case number:
//   - "reshape_target" (+ "special_zero"): an explicit pattern, with -1 on the runtime (token) axis and 0
//     copying an input dimension, so the same pattern serves prefill and decode;
//   - a second input with "shape_axes": dimensions copied from that reference tensor (-1 entries in
//     "shape_axes" keep the constant from "reshape_target"), which preserves variable image grids;
//   - a second input without "shape_axes": the runtime pattern itself;
//   - "merge_heads": [B, T, H, S] -> [B, 1, T, H*S] after attention.
OutputVector translate_reshape(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    const auto input = context.get_input(0);
    const auto target = context.get_attribute<std::vector<int64_t>>("reshape_target", {});
    if (context.get_input_size() == 2) {
        const auto axes = context.get_attribute<std::vector<int64_t>>("shape_axes", {});
        if (axes.empty()) {
            return rename_outputs_with_suffix(
                {std::make_shared<ov::op::v1::Reshape>(input,
                                                       context.get_input(1),
                                                       context.get_attribute<bool>("special_zero", false))},
                context.get_name());
        }
        FRONT_END_OP_CONVERSION_CHECK(target.size() == axes.size(), "Invalid reference reshape pattern");
        // One ShapeOf for the whole pattern, and one Gather per run of copied axes.
        const auto reference_shape = std::make_shared<ov::op::v3::ShapeOf>(context.get_input(1), ov::element::i64);
        ov::OutputVector dimensions;
        for (size_t i = 0; i < target.size();) {
            if (axes[i] < 0) {
                dimensions.push_back(ov::op::v0::Constant::create(ov::element::i64, {1}, {target[i]}));
                ++i;
                continue;
            }
            std::vector<int> copied;
            for (; i < target.size() && axes[i] >= 0; ++i)
                copied.push_back(static_cast<int>(axes[i]));
            dimensions.push_back(gather_dims(reference_shape, copied));
        }
        return rename_outputs_with_suffix(
            {std::make_shared<ov::op::v1::Reshape>(input, std::make_shared<ov::op::v0::Concat>(dimensions, 0), false)},
            context.get_name());
    }
    if (!target.empty()) {
        auto pattern = ov::op::v0::Constant::create(ov::element::i64, {target.size()}, target);
        return rename_outputs_with_suffix(
            {std::make_shared<ov::op::v1::Reshape>(input, pattern, context.get_attribute<bool>("special_zero", false))},
            context.get_name());
    }
    FRONT_END_OP_CONVERSION_CHECK(context.get_attribute<bool>("merge_heads", false),
                                  "RESHAPE requires a pattern input, \"reshape_target\" or \"merge_heads\"");
    const auto shape = input.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(shape.rank() == 4, "Head merging requires a rank-4 input");
    ov::Output<ov::Node> pattern;
    if (shape[2].is_static() && shape[3].is_static()) {
        pattern = ov::op::v0::Constant::create(ov::element::i64,
                                               {4},
                                               std::vector<int64_t>{0, 1, -1, shape[2].get_length() * shape[3].get_length()});
    } else {
        auto dimensions = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
        auto heads = std::make_shared<ov::op::v8::Gather>(dimensions,
                                                          ov::op::v0::Constant::create(ov::element::i64, {2}, {2, 3}),
                                                          ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        auto width = std::make_shared<ov::op::v1::ReduceProd>(heads,
                                                              ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                                                              true);
        pattern = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, -1}), width},
            0);
    }
    return rename_outputs_with_suffix({std::make_shared<ov::op::v1::Reshape>(input, pattern, true)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
