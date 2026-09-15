// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_attention_onnx.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils/attention.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft {

namespace detail {
using ov::frontend::onnx::attention::get_dimensions;
}  // namespace detail

namespace opset_1 {
ov::OutputVector paged_attention(const ov::frontend::onnx::Node& node) {
    // com.microsoft.PagedAttention (ONNX Runtime) inputs, in order:
    //   0 query (required), 1 key, 2 value (both absent in packed-QKV mode), 3 key_cache, 4 value_cache,
    //   5 cumulative_sequence_length, 6 past_seqlens, 7 block_table, 8 cos_cache, 9 sin_cache (rotary only).
    // 8 inputs are always present; key/value (1/2) are empty strings in packed mode and cos/sin (8/9) are
    // present only when do_rotary is set.
    constexpr size_t inputs_count_min = 8;
    constexpr size_t inputs_count_max = 10;
    common::default_op_checks(node, inputs_count_min, inputs_count_max);

    const auto onnx_op_inputs = node.get_ov_inputs();
    const auto num_heads = node.get_attribute_value<int64_t>("num_heads");
    const auto kv_num_heads = node.get_attribute_value<int64_t>("kv_num_heads");
    const auto scale = node.get_attribute_value<float>("scale", 0.0f);
    const auto softcap = node.get_attribute_value<float>("softcap", 0.0f);
    const auto local_window_size = node.get_attribute_value<int64_t>("local_window_size", -1);
    const auto do_rotary = node.get_attribute_value<int64_t>("do_rotary", 0);
    const auto rotary_interleaved = node.get_attribute_value<int64_t>("rotary_interleaved", 0);

    // Reject attributes whose semantics are not implemented by this decomposition. local_window_size == -1
    // disables the window; >= 1 selects a sliding window. A window of size 0 is an empty attention (every
    // query masks all keys, including its own diagonal) and is not a valid ONNX Runtime config; rejecting it
    // keeps the finite-lowest() mask NaN-safe (every row keeps its own diagonal key).
    FRONT_END_OP_CONVERSION_CHECK(local_window_size == -1 || local_window_size >= 1,
                                  "PagedAttention: local_window_size must be -1 (disabled) or >= 1, got ",
                                  local_window_size,
                                  ".");
    // Rotary requires cos_cache (input 8) and sin_cache (input 9).
    if (0 != do_rotary) {
        constexpr size_t cos_cache_index = 8;
        constexpr size_t sin_cache_index = 9;
        FRONT_END_OP_CONVERSION_CHECK(common::is_input_valid(onnx_op_inputs, cos_cache_index) &&
                                          common::is_input_valid(onnx_op_inputs, sin_cache_index),
                                      "PagedAttention: cos_cache and sin_cache inputs are required when "
                                      "do_rotary is enabled.");
    }

    auto Q = onnx_op_inputs[0];
    const auto& K = onnx_op_inputs[1];
    const auto& V = onnx_op_inputs[2];

    FRONT_END_OP_CONVERSION_CHECK(!ov::op::util::is_null(Q), "PagedAttention: query/QKV must not be null.");

    // Query and key/value are 2-D packed-varlen tensors [num_tokens, hidden]. The internal op consumes three
    // separate 2-D activation tensors, so split the packed-QKV form (key/value absent) into Q, K, V along the
    // hidden dimension: [num_tokens, (num_heads + 2 * kv_num_heads) * head_size] -> Q + K + V.
    ov::OutputVector ov_op_inputs;
    if (ov::op::util::is_null(K) && ov::op::util::is_null(V)) {
        const auto q_shape_node = std::make_shared<v3::ShapeOf>(Q);
        const auto hidden_size_node = detail::get_dimensions(q_shape_node, {1});
        const auto total_num_heads_node =
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads + kv_num_heads + kv_num_heads});
        const auto head_size_node = std::make_shared<v1::Divide>(hidden_size_node, total_num_heads_node);
        // hidden feature counts of the Q / K / V slices: [num_heads * head, kv_num_heads * head, kv_num_heads * head].
        const auto q_hidden =
            std::make_shared<v1::Multiply>(head_size_node,
                                           v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads}));
        const auto kv_hidden =
            std::make_shared<v1::Multiply>(head_size_node,
                                           v0::Constant::create(ov::element::i64, ov::Shape{1}, {kv_num_heads}));
        const auto split_lengths = std::make_shared<v0::Concat>(ov::NodeVector{q_hidden, kv_hidden, kv_hidden}, 0);
        const auto split_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        const auto split = std::make_shared<v1::VariadicSplit>(Q, split_axis, split_lengths);
        FRONT_END_OP_CONVERSION_CHECK(split->get_output_size() == 3,
                                      "PagedAttention: Expecting packed QKV split to produce 3 outputs.");
        ov_op_inputs.push_back(split->output(0));
        ov_op_inputs.push_back(split->output(1));
        ov_op_inputs.push_back(split->output(2));
    } else {
        FRONT_END_OP_CONVERSION_CHECK(!ov::op::util::is_null(K), "PagedAttention: key must not be null.");
        FRONT_END_OP_CONVERSION_CHECK(!ov::op::util::is_null(V), "PagedAttention: value must not be null.");
        ov_op_inputs.push_back(Q);
        ov_op_inputs.push_back(K);
        ov_op_inputs.push_back(V);
    }

    // Append the fixed required inputs: key_cache(3), value_cache(4), cumulative_sequence_length(5),
    // past_seqlens(6), block_table(7). ONNX-specific optional-input handling stays entirely inside the
    // frontend: the internal op is given a clean, fixed arity (8 without rotary, 10 with) and never receives
    // a NullNode placeholder. ONNX Runtime always emits 10 input slots, filling absent optionals with empty
    // strings (which arrive as null here), so cos_cache(8)/sin_cache(9) are appended only when do_rotary is
    // set (validated above); they are otherwise dropped rather than forwarded as nulls.
    for (size_t i = 3; i <= 7; ++i) {
        FRONT_END_OP_CONVERSION_CHECK(!ov::op::util::is_null(onnx_op_inputs[i]),
                                      "PagedAttention: required input ",
                                      i,
                                      " (key_cache/value_cache/cumulative_sequence_length/past_seqlens/"
                                      "block_table) must not be null.");
        ov_op_inputs.push_back(onnx_op_inputs[i]);
    }
    if (0 != do_rotary) {
        ov_op_inputs.push_back(onnx_op_inputs[8]);
        ov_op_inputs.push_back(onnx_op_inputs[9]);
    }

    return std::make_shared<internal::PagedAttentionONNX>(ov_op_inputs,
                                                          num_heads,
                                                          kv_num_heads,
                                                          scale,
                                                          softcap,
                                                          local_window_size,
                                                          do_rotary != 0,
                                                          rotary_interleaved != 0)
        ->outputs();
}

ONNX_OP("PagedAttention", OPSET_SINCE(1), com_microsoft::opset_1::paged_attention, MICROSOFT_DOMAIN);

}  // namespace opset_1

}  // namespace ov::frontend::onnx::com_microsoft
