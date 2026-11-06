// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softmax.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

static bool is_static_one(const ov::Dimension& dim) {
    return dim.is_static() && dim.get_length() == 1;
}

static bool same_static_dim(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    return lhs.is_static() && rhs.is_static() && lhs.get_length() == rhs.get_length();
}

// Attention sinks (gpt-oss): a per-head learned bias appended as one hidden logit column before the
// softmax and dropped afterwards. The sinks tensor is [1,1,1,n_head] (matches the logits' head dim).
static bool is_attention_sinks_input_shape(const ov::PartialShape& candidate, const ov::PartialShape& logits_shape) {
    if (candidate.rank().is_dynamic() || logits_shape.rank().is_dynamic() || candidate.rank().get_length() != 4 ||
        logits_shape.rank().get_length() != 4) {
        return false;
    }
    return is_static_one(candidate[0]) && is_static_one(candidate[1]) && is_static_one(candidate[2]) &&
           same_static_dim(candidate[3], logits_shape[1]);
}

// Append attention sinks as one hidden logit column, softmax over the last axis, then drop it.
static ov::Output<ov::Node> apply_sinks(const NodeContext& context,
                                        const ov::Output<ov::Node>& logits,
                                        const ov::Output<ov::Node>& sinks_in) {
    ov::Output<ov::Node> sinks = sinks_in;
    if (sinks.get_element_type() != logits.get_element_type()) {
        sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
    }
    auto sink_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, -1, 1, 1});
    auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sink_shape, false);

    auto logits_shape = std::make_shared<ov::op::v3::ShapeOf>(logits, ov::element::i64);
    auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
    auto four = ov::op::v0::Constant::create(ov::element::i64, {1}, {4});
    auto shape_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});

    auto sink_prefix_shape = std::make_shared<ov::op::v8::Slice>(logits_shape, zero, three, one, shape_axis);
    auto sink_last_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto sink_broadcast_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sink_prefix_shape, sink_last_dim}, 0);
    auto sink_column =
        std::make_shared<ov::op::v3::Broadcast>(sinks_4d, sink_broadcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
    auto softmax_input = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{logits, sink_column}, 3);
    auto softmax_with_sink = std::make_shared<ov::op::v8::Softmax>(softmax_input, -1);
    auto original_last_dim = std::make_shared<ov::op::v8::Slice>(logits_shape, three, four, one, shape_axis);
    return std::make_shared<ov::op::v8::Slice>(softmax_with_sink, zero, original_last_dim, one, three);
}

OutputVector translate_soft_max(const NodeContext& context) {
    num_inputs_check(context, 1, 3);

    auto input0 = context.get_input(0);
    ov::Output<Node> res;

    // ggml SOFT_MAX reduces ne[0] == the OV last axis (rank-3 attention -> 2, rank-4 router -> 3);
    // derive it from the rank instead of hardcoding 2. The native builder may request another
    // axis, e.g. the gpt-oss top-k gating weights [1,T,K,1] normalize over K. Reused by both
    // Softmax paths below.
    const auto& in_ps = input0.get_partial_shape();
    const bool has_softmax_axis = context.has_attribute("softmax_axis");
    const int64_t softmax_axis = has_softmax_axis ? context.get_attribute<int64_t>("softmax_axis")
                                                  : (in_ps.rank().is_static() ? in_ps.rank().get_length() - 1 : 2);
    if (has_softmax_axis && in_ps.rank().is_static()) {
        const int64_t rank = in_ps.rank().get_length();
        FRONT_END_OP_CONVERSION_CHECK(-rank <= softmax_axis && softmax_axis < rank,
                                      "SOFT_MAX softmax_axis ",
                                      softmax_axis,
                                      " is out of range for input rank ",
                                      rank);
    }

    float scale = context.get_attribute<float>("scale", 1.0f);
    float max_bias = context.get_attribute<float>("max_bias", 0.0f);

    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
    ov::Output<ov::Node> scaled_input = std::make_shared<ov::op::v1::Multiply>(input0, scale_node);

    // Disambiguate a 2nd input: it is either the additive mask or (gpt-oss) the attention sinks.
    const bool second_input_is_sinks =
        context.get_input_size() == 2 &&
        is_attention_sinks_input_shape(context.get_input_shape(1), context.get_input_shape(0));
    const bool has_mask = context.get_input_size() > 1 && !second_input_is_sinks;
    const bool has_sinks = second_input_is_sinks || context.get_input_size() > 2;
    const int sinks_input_idx = second_input_is_sinks ? 1 : 2;
    if (has_sinks && has_softmax_axis) {
        const bool last_axis =
            softmax_axis == -1 || (in_ps.rank().is_static() && softmax_axis == in_ps.rank().get_length() - 1);
        FRONT_END_OP_CONVERSION_CHECK(last_axis, "SOFT_MAX with attention sinks supports only the last softmax_axis");
    }

    if (!has_mask) {
        if (has_sinks) {
            res = apply_sinks(context, scaled_input, context.get_input(sinks_input_idx));
            return rename_outputs_with_suffix({std::move(res)}, context.get_name());
        }
        res = std::make_shared<ov::op::v8::Softmax>(scaled_input, softmax_axis);
        return rename_outputs_with_suffix({std::move(res)}, context.get_name());
    }

    ov::Output<ov::Node> mask_node_sliced;
    if (context.has_input("KQ_mask_sliced")) {
        mask_node_sliced = context.get_input("KQ_mask_sliced");
    } else {
        auto mask_node = context.get_input(1);
        const auto rank = mask_node.get_partial_shape().rank();
        FRONT_END_OP_CONVERSION_CHECK(rank.is_static() && rank.get_length() >= 2,
                                      "SOFT_MAX mask requires rank at least two");
        const int axis = static_cast<int>(rank.get_length()) - 2;
        mask_node_sliced = mask_node;
        if (mask_node.get_partial_shape()[axis] != 1) {
            auto token_len = get_dimensions(input0, {-2});
            auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            mask_node_sliced =
                std::make_shared<ov::op::v8::Slice>(mask_node,
                                                    zero,
                                                    token_len,
                                                    one,
                                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {axis}));
        }
    }

    auto output_type = input0.get_element_type();
    if (mask_node_sliced.get_element_type() != output_type) {
        mask_node_sliced = std::make_shared<ov::op::v0::Convert>(mask_node_sliced, output_type);
    }

    ov::Output<ov::Node> biased_input = scaled_input;
    if (max_bias > 0.0f) {
        // ALiBi: per-head slope[h] applied to the mask (ggml ggml_compute_forward_soft_max_f32).
        // ggml's head dimension is always ne[2] regardless of rank; reversing ggml's dims to OV
        // order puts it at (rank - 3) -- axis 1 for the usual 4D [batch, head, token, kv] layout,
        // axis 0 when a leading batch/repetition dim of 1 is omitted (rank 3).
        FRONT_END_OP_CONVERSION_CHECK(in_ps.rank().is_static() && in_ps.rank().get_length() >= 3,
                                      "SOFT_MAX ALiBi requires a static rank of at least 3");
        const int64_t rank = in_ps.rank().get_length();
        const int64_t head_axis = rank - 3;
        FRONT_END_OP_CONVERSION_CHECK(in_ps[head_axis].is_static(),
                                      "SOFT_MAX ALiBi requires a static head-count dimension");
        const auto slope_node = make_alibi_slopes(output_type,
                                                  static_cast<size_t>(rank),
                                                  static_cast<size_t>(head_axis),
                                                  static_cast<uint32_t>(in_ps[head_axis].get_length()),
                                                  max_bias);
        auto slope_mask = std::make_shared<ov::op::v1::Multiply>(mask_node_sliced, slope_node);
        biased_input = std::make_shared<ov::op::v1::Add>(scaled_input, slope_mask);
    } else {
        biased_input = std::make_shared<ov::op::v1::Add>(scaled_input, mask_node_sliced);
    }

    if (has_sinks) {
        res = apply_sinks(context, biased_input, context.get_input(sinks_input_idx));
        return rename_outputs_with_suffix({std::move(res)}, context.get_name());
    }

    res = std::make_shared<ov::op::v8::Softmax>(biased_input, softmax_axis);

    return rename_outputs_with_suffix({std::move(res)}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
