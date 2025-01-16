// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

std::shared_ptr<ov::Node> get_present_state(const std::shared_ptr<ov::Node>& K,
                                            const std::shared_ptr<ov::Node>& V,
                                            const ov::OutputVector& op_inputs);
std::shared_ptr<ov::Node> rotaryEmbedding(ov::Output<ov::Node> input,
                                          ov::Output<ov::Node> past_seqlen,
                                          std::shared_ptr<ov::Node> seqlen_k,
                                          std::shared_ptr<ov::Node> cos_cache,
                                          std::shared_ptr<ov::Node> sin_cache,
                                          std::shared_ptr<ov::Node> dim_head_size,
                                          bool interleaved);
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims);
ov::OutputVector make_split(const ov::Output<ov::Node>& value, int64_t num_splits, int64_t axis);
ov::OutputVector make_split(const ov::Output<ov::Node>& value, const std::vector<int64_t>& split_lengths, int64_t axis);

ov::pass::GroupQueryAttentionDecomposition::GroupQueryAttentionDecomposition() {
    MATCHER_SCOPE(GroupQeuryAttentionDecomposition);
    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::GroupQueryAttention>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node =
            ov::as_type_ptr<ov::op::GroupQueryAttention>(pattern_to_output.at(pattern_node).get_node_shared_ptr());

        if (node == nullptr || transformation_callback(node)) {
            return false;
        }

        auto new_output_node = decompose(node);
        ov::replace_node(node, new_output_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, matcher_name);
    register_matcher(m, callback);
}

ov::OutputVector ov::pass::GroupQueryAttentionDecomposition::decompose(
    std::shared_ptr<ov::op::GroupQueryAttention> node) {
    using namespace ov::op;

    const auto num_heads = node->get_num_heads();
    const auto kv_num_heads = node->get_kv_num_heads();
    const auto scale = node->get_scale();
    const auto do_rotary = node->get_do_rotary();
    const auto rotary_interleaved = node->get_rotary_interleaved();
    // TODO: add softcap support

    auto Q = node->input_value(0);
    auto K = node->input_value(1);
    auto V = node->input_value(2);
    auto past_key = node->input_value(3);
    auto past_value = node->input_value(4);
    auto seqlens_k = node->input_value(5);
    auto total_sequence_length = node->input_value(6);  // unused, it's not always equal (seqlens_k + 1)
    auto cos_cache = node->input_value(7);
    auto sin_cache = node->input_value(8);

    const auto node_shape = std::make_shared<v3::ShapeOf>(Q);
    const auto batch_size = get_dimensions(node_shape, {0});
    const auto current_seqlen_size = get_dimensions(node_shape, {1});
    const auto hidden_size = get_dimensions(node_shape, {2});
    const auto total_num_heads_node =
        v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads + kv_num_heads + kv_num_heads});
    auto head_size_node = std::make_shared<v1::Divide>(hidden_size, total_num_heads_node);

    // transpose Q, K and V to (batch_size, num_heads, sequence_len, head_size)
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    if (GroupQueryAttention::is_null(K)) {
        // Handle the packed QKV
        auto packed_qkv_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size, current_seqlen_size, total_num_heads_node, head_size_node},
            0);
        auto inputs_qkv = std::make_shared<v1::Reshape>(Q, packed_qkv_shape, false)->output(0);
        // (batch_size, sequence_len, num_head, head_size)
        inputs_qkv = std::make_shared<v1::Transpose>(inputs_qkv, perm);
        // split the node into 3 even parts Q, K, V with shape (batch_size, num_head, sequence_len, head_size)
        auto split = make_split(inputs_qkv, {num_heads, kv_num_heads, kv_num_heads}, 1);
        Q = split[0];
        K = split[1];
        V = split[2];
    } else {
        Q = std::make_shared<v1::Transpose>(Q, perm);
        K = std::make_shared<v1::Transpose>(K, perm);
        V = std::make_shared<v1::Transpose>(V, perm);
    }

    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto one_without_shape = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto seqlens_elemi64 = std::make_shared<v0::Convert>(seqlens_k, ov::element::i64);
    auto real_seqlens = std::make_shared<v1::Add>(seqlens_elemi64, one);

    // Only consider batch is 1
    auto seqlens_1d = std::make_shared<v1::Reshape>(real_seqlens, one, false);
    auto past_sequence_length = std::make_shared<v1::Subtract>(seqlens_1d, current_seqlen_size);

    if (do_rotary) {
        Q = rotaryEmbedding(Q,
                            past_sequence_length,
                            seqlens_1d,
                            cos_cache.get_node_shared_ptr(),
                            sin_cache.get_node_shared_ptr(),
                            head_size_node,
                            rotary_interleaved);
        K = rotaryEmbedding(K,
                            past_sequence_length,
                            seqlens_1d,
                            cos_cache.get_node_shared_ptr(),
                            sin_cache.get_node_shared_ptr(),
                            head_size_node,
                            rotary_interleaved);
    }
    // present = concat(K, V) if 'past' input is unavailable
    // or
    // present = concat(past, K, V)
    auto construct_kv_cache = [&](const ov::Output<ov::Node>& past, const ov::Output<ov::Node>& current) {
        auto past_datas = std::make_shared<v8::Slice>(past, zero, past_sequence_length, one, two);
        auto curr_datas = std::make_shared<v8::Slice>(current, zero, current_seqlen_size, one, two);
        return std::make_shared<v0::Concat>(ov::NodeVector{past_datas, curr_datas}, 2);
    };

    K = construct_kv_cache(past_key, K);
    V = construct_kv_cache(past_value, V);
    auto present_k = K.get_node_shared_ptr();
    auto present_v = V.get_node_shared_ptr();

    const size_t kv_num_heads_factor = num_heads / kv_num_heads;
    if (kv_num_heads_factor > 1) {
        const auto kv_shape = std::make_shared<v3::ShapeOf>(K);
        // (batch_size, num_heads, sequence_len, head_size)
        const auto kv_shape_prev_2 = get_dimensions(kv_shape, {0, 1});
        const auto kv_shape_last_2 = get_dimensions(kv_shape, {2, 3});
        auto new_kv_shape = std::make_shared<v0::Concat>(ov::NodeVector{kv_shape_prev_2, one, kv_shape_last_2}, 0);
        K = std::make_shared<v1::Reshape>(K, new_kv_shape, false);
        V = std::make_shared<v1::Reshape>(V, new_kv_shape, false);
        K = std::make_shared<v0::Concat>(ov::OutputVector(kv_num_heads_factor, K), 2);
        V = std::make_shared<v0::Concat>(ov::OutputVector(kv_num_heads_factor, V), 2);
        auto q_shape = std::make_shared<v3::ShapeOf>(Q);
        // (batch_size, num_heads, sequence_len, head_size)
        const auto q_shape_prev_2 = get_dimensions(q_shape, {0, 1});
        auto extended_kv_shape = std::make_shared<v0::Concat>(ov::NodeVector{q_shape_prev_2, kv_shape_last_2}, 0);
        K = std::make_shared<v1::Reshape>(K, extended_kv_shape, false);
        V = std::make_shared<v1::Reshape>(V, extended_kv_shape, false);
    }

    // need to apply low-triangle mask to attention score.
    // two steps, construct the total_sequence x total_sequence triangle, then slice the current length
    auto seqlens_1d_scalar = std::make_shared<v1::Reshape>(seqlens_1d, one_without_shape, false);  // 12 or 13
    std::shared_ptr<ov::Node> mask_per_line_node =
        std::make_shared<v4::Range>(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                    seqlens_1d_scalar,
                                    one_without_shape,
                                    ov::element::i64);                            // [0,1,2,...,]
    auto hori_range = std::make_shared<v0::Unsqueeze>(mask_per_line_node, zero);  // 1x12 or 1x13
    auto vert_range = std::make_shared<v0::Unsqueeze>(mask_per_line_node, one);   // 12x1 or 13x1
    auto triu = std::make_shared<v1::Greater>(hori_range, vert_range);            // 12x12 or 13x13
    auto typed_zero = v0::Constant::create(ov::element::f32, ov::Shape{}, {0});
    auto minus_inf = v0::Constant::create(ov::element::f32, ov::Shape{}, {-std::numeric_limits<float>::infinity()});
    auto atten_mask = std::make_shared<v1::Select>(triu, minus_inf, typed_zero);  // 12x12 or 13x13
    auto atten_mask_sliced = std::make_shared<v8::Slice>(atten_mask,
                                                         past_sequence_length,
                                                         seqlens_1d,
                                                         one,
                                                         zero);  // slice to current query seqlen, 12x12 or 1x13

    // compute softmax((Q x K') / sqrt(head_size)) x V
    std::shared_ptr<ov::Node> qga_output;
    if (scale != 0.0f) {
        auto scale_node = v0::Constant::create(ov::element::f32, Shape{}, {scale});
        qga_output = std::make_shared<v13::ScaledDotProductAttention>(Q, K, V, atten_mask_sliced, scale_node, false);
    } else {
        qga_output = std::make_shared<v13::ScaledDotProductAttention>(Q, K, V, atten_mask_sliced, false);
    }

    // transpose the result from (batch_size, num_heads, sequence_length, head_size)
    // to (batch_size, sequence_length, num_heads, head_size)
    auto qga_output_transposed = std::make_shared<v1::Transpose>(qga_output, perm);
    auto dim_merge_shape = v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1});
    // reshape the result from (batch_size, sequence_length, num_heads, head_size)
    //          to             (batch_size, sequence_length, num_heads * head_size)
    auto output = std::make_shared<v1::Reshape>(qga_output_transposed, dim_merge_shape, true);

    return {output, present_k, present_v};
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims) {
    using namespace ov::op;
    static const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims) {
    return get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(node), dims);
}

std::shared_ptr<ov::Node> rotaryEmbedding(ov::Output<ov::Node> input,
                                          ov::Output<ov::Node> past_seqlen,
                                          std::shared_ptr<ov::Node> seqlen_k,
                                          std::shared_ptr<ov::Node> cos_cache,
                                          std::shared_ptr<ov::Node> sin_cache,
                                          std::shared_ptr<ov::Node> dim_head_size,
                                          bool interleaved) {
    using namespace ov::op;
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    auto slice_cache_dim_shape = seqlen_k;

    auto cos = std::make_shared<v8::Slice>(cos_cache, past_seqlen, slice_cache_dim_shape, one, zero);
    auto sin = std::make_shared<v8::Slice>(sin_cache, past_seqlen, slice_cache_dim_shape, one, zero);

    if (interleaved) {
        auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

        auto cache_shape = std::make_shared<v3::ShapeOf>(cos_cache);
        auto cache_last_dim = get_dimensions(cos_cache, {-1});

        auto input_shape = std::make_shared<v3::ShapeOf>(input);

        auto dim_bns = get_dimensions(input_shape, {0, 1, 2});
        std::shared_ptr<ov::Node> half_last_dim = cache_last_dim;

        auto negtive_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto split_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, half_last_dim, two}, 0);
        auto reshaped_input = std::make_shared<v1::Reshape>(input, split_input_shape, false);

        auto in_split = make_split(reshaped_input, 2, -1);
        auto res_0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(in_split[0], cos),
                                                    std::make_shared<v1::Multiply>(in_split[1], sin));
        auto res_1 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(in_split[0], sin),
                                               std::make_shared<v1::Multiply>(in_split[1], cos));

        auto concat_ret = std::make_shared<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);
        return std::make_shared<v1::Reshape>(concat_ret, input_shape, false);
    } else {
        auto in_split = make_split(input, 2, -1);
        auto res_0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(in_split[0], cos),
                                                    std::make_shared<v1::Multiply>(in_split[1], sin));
        auto res_1 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(in_split[0], sin),
                                               std::make_shared<v1::Multiply>(in_split[1], cos));

        return std::make_shared<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);
    }
}

// make split functions is a copy-past from ONNX FE. TODO: move it to one place
ov::OutputVector make_split(const ov::Output<ov::Node>& value,
                            const std::vector<int64_t>& split_lengths,
                            int64_t axis) {
    using namespace ov::op;
    const auto axis_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
    const auto split_lengths_node =
        v0::Constant::create(ov::element::i64, ov::Shape{split_lengths.size()}, split_lengths);
    const auto variadic_split = std::make_shared<v1::VariadicSplit>(value, axis_node, split_lengths_node);

    return variadic_split->outputs();
}

ov::OutputVector make_split(const ov::Output<ov::Node>& value, int64_t num_splits, int64_t axis) {
    using namespace ov::op;
    const auto axis_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
    const auto split = std::make_shared<v1::Split>(value, axis_node, num_splits);

    return split->outputs();
}