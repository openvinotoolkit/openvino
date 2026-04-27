// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/decompositions/rope.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_23 {

namespace {

// Build a 1-D i64 Constant from a list of values. We use std::vector explicitly so
// that mixed `int64_t` / `int` literals don't break Constant::create template deduction.
std::shared_ptr<v0::Constant> i64_const(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

// Reshape with `special_zero=true` so a literal 0 in `target` means "keep the
// corresponding input dimension" — saves us from gathering shape parts dynamically.
std::shared_ptr<ov::Node> reshape_keep(const ov::Output<ov::Node>& x, const std::vector<int64_t>& target) {
    return std::make_shared<v1::Reshape>(x, i64_const(target), /*special_zero=*/true);
}

}  // namespace

ov::OutputVector rotary_embedding(const ov::frontend::onnx::Node& node) {
    // Operator definition: https://onnx.ai/onnx/operators/onnx__RotaryEmbedding.html
    //
    // Strategy: normalize layout to [bs, num_heads, seq, head_size], optionally peel off
    // the no-rotate tail of the head dim, optionally de-interleave for the interleaved
    // mode, delegate the core formula to ov::decompositions::rope (recognised by
    // ov::pass::RoPEFusion), then undo the wrappers.
    //
    // Note: we never need to know `head_size` statically — the rope helper's split is
    // sized from `half_rotary_dim` (derived from `rotary_embedding_dim` or cos_cache),
    // and partial-rotation tail length uses VariadicSplit's `-1` (take-the-rest) form.
    const auto inputs = node.get_ov_inputs();
    CHECK_VALID_NODE(node,
                     inputs.size() == 3 || inputs.size() == 4,
                     "RotaryEmbedding expects 3 or 4 inputs (X, cos_cache, sin_cache, [position_ids]). Got: ",
                     inputs.size());

    const auto& input = inputs[0];
    const auto& cos_cache = inputs[1];
    const auto& sin_cache = inputs[2];
    const bool has_position_ids = inputs.size() == 4 && !ov::op::util::is_null(inputs[3]);

    const int64_t interleaved = node.get_attribute_value<int64_t>("interleaved", 0);
    const int64_t rotary_dim = node.get_attribute_value<int64_t>("rotary_embedding_dim", 0);
    const int64_t num_heads = node.get_attribute_value<int64_t>("num_heads", 0);
    const bool partial = rotary_dim > 0;
    if (partial) {
        CHECK_VALID_NODE(node, rotary_dim % 2 == 0, "rotary_embedding_dim must be even, got: ", rotary_dim);
    }

    // The shared rope helper requires a static `half_rotary_dim` to build a VariadicSplit.
    const auto& cos_pshape = cos_cache.get_partial_shape();
    CHECK_VALID_NODE(node,
                     cos_pshape.rank().is_static() && cos_pshape.rank().get_length() >= 2 &&
                         cos_pshape[cos_pshape.rank().get_length() - 1].is_static(),
                     "cos_cache must have static rank >= 2 and a static last dimension, got: ",
                     cos_pshape);
    const int64_t cos_last_dim = cos_pshape[cos_pshape.rank().get_length() - 1].get_length();
    const int64_t half_rotary_dim = partial ? rotary_dim / 2 : cos_last_dim;

    const auto input_rank = input.get_partial_shape().rank();
    CHECK_VALID_NODE(node,
                     input_rank.is_static() && (input_rank.get_length() == 3 || input_rank.get_length() == 4),
                     "RotaryEmbedding input must have static rank 3 or 4.");
    const bool input_is_3d = input_rank.get_length() == 3;
    if (input_is_3d) {
        CHECK_VALID_NODE(node, num_heads > 0, "num_heads attribute is required for 3D input.");
    }

    // ---- Step 1. Normalize layout to [bs, num_heads, seq, head_size] ----
    const auto perm_bnsh = i64_const({0, 2, 1, 3});
    const auto input_shape = input_is_3d ? std::make_shared<v3::ShapeOf>(input, ov::element::i64) : nullptr;

    ov::Output<ov::Node> x = input;
    if (input_is_3d) {
        // [bs, seq, hidden] -> [bs, seq, num_heads, head_size] -> [bs, num_heads, seq, head_size]
        x = reshape_keep(input, {0, 0, num_heads, -1});
        x = std::make_shared<v1::Transpose>(x, perm_bnsh);
    }

    // ---- Step 2. For partial rotation, peel off the no-rotate tail of the head dim ----
    ov::Output<ov::Node> x_rotate = x;
    ov::Output<ov::Node> x_no_rotate;
    if (partial) {
        const auto axis_neg1_scalar = v0::Constant::create(ov::element::i64, ov::Shape{}, {int64_t{-1}});
        // {rotary_dim, -1}: -1 = "the rest". When rotary_dim == head_size, the second piece is empty.
        const auto lengths = i64_const({rotary_dim, -1});
        auto split = std::make_shared<v1::VariadicSplit>(x, axis_neg1_scalar, lengths);
        x_rotate = split->output(0);
        x_no_rotate = split->output(1);
    }

    // ---- Step 3. Build cos/sin in shape [..., 1, ..., half_rotary_dim] ----
    ov::Output<ov::Node> cos = cos_cache;
    ov::Output<ov::Node> sin = sin_cache;
    if (has_position_ids) {
        const auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {int64_t{0}});
        cos = std::make_shared<v8::Gather>(cos_cache, inputs[3], axis0);
        sin = std::make_shared<v8::Gather>(sin_cache, inputs[3], axis0);
    }
    // For partial rotation, the cache may hold the full head_size/2; slice it down.
    if (cos_last_dim > half_rotary_dim) {
        cos = std::make_shared<v8::Slice>(cos,
                                          i64_const({0}),
                                          i64_const({half_rotary_dim}),
                                          i64_const({1}),
                                          i64_const({-1}));
        sin = std::make_shared<v8::Slice>(sin,
                                          i64_const({0}),
                                          i64_const({half_rotary_dim}),
                                          i64_const({1}),
                                          i64_const({-1}));
    }
    // Insert axis=1 broadcast dim → [bs, 1, seq, half_rotary_dim].
    const auto axis_1_1d = i64_const({1});
    cos = std::make_shared<v0::Unsqueeze>(cos, axis_1_1d);
    sin = std::make_shared<v0::Unsqueeze>(sin, axis_1_1d);

    // ---- Step 4. Optionally de-interleave so the split-half formula in `rope` applies ----
    // [.., rotary_dim] -reshape-> [.., half, 2] -transpose [0,1,2,4,3]-> [.., 2, half] -reshape-> [.., rotary_dim]
    const auto perm_5d = i64_const({0, 1, 2, 4, 3});
    ov::Output<ov::Node> rope_input = x_rotate;
    if (interleaved) {
        rope_input = reshape_keep(x_rotate, {0, 0, 0, half_rotary_dim, 2});
        rope_input = std::make_shared<v1::Transpose>(rope_input, perm_5d);
        rope_input = reshape_keep(rope_input, {0, 0, 0, -1});
    }

    // ---- Step 5. Core RoPE formula via the shared decomposition helper ----
    ov::pass::NodeRegistry rope_reg;
    ov::Output<ov::Node> rotated = ov::decompositions::rope(rope_reg, rope_input, cos, sin, half_rotary_dim);

    // ---- Step 6. Re-interleave if needed (mirror of step 4) ----
    if (interleaved) {
        rotated = reshape_keep(rotated, {0, 0, 0, 2, half_rotary_dim});
        rotated = std::make_shared<v1::Transpose>(rotated, perm_5d);
        rotated = reshape_keep(rotated, {0, 0, 0, -1});
    }

    // ---- Step 7. Concat rotated and no-rotate parts back along the head dim ----
    ov::Output<ov::Node> output =
        partial ? std::make_shared<v0::Concat>(ov::OutputVector{rotated, x_no_rotate}, -1) : rotated;

    // ---- Step 8. Restore original 3D layout if needed ----
    if (input_is_3d) {
        // [bs, num_heads, seq, head_size] -> [bs, seq, num_heads, head_size] -> [bs, seq, hidden]
        output = std::make_shared<v1::Transpose>(output, perm_bnsh);
        output = std::make_shared<v1::Reshape>(output, input_shape, /*special_zero=*/false);
    }

    return {output};
}

ONNX_OP("RotaryEmbedding", OPSET_SINCE(1), ai_onnx::opset_23::rotary_embedding);

}  // namespace opset_23
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
