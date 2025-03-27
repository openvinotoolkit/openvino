// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::GroupQueryAttentionDecomposition::GroupQueryAttentionDecomposition() {
    MATCHER_SCOPE(GroupQeuryAttentionDecomposition);
    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::internal::GroupQueryAttention>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node = ov::as_type_ptr<ov::op::internal::GroupQueryAttention>(
            pattern_to_output.at(pattern_node).get_node_shared_ptr());

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
    std::shared_ptr<ov::op::internal::GroupQueryAttention> node) {
    using namespace ov::op;

    const auto num_heads = node->get_num_heads();
    const auto kv_num_heads = node->get_kv_num_heads();
    const auto scale = node->get_scale();
    const auto do_rotary = node->get_do_rotary();
    const auto rotary_interleaved = node->get_rotary_interleaved();
    const auto is_static_input = node->get_is_static_input();
    // TODO: add softcap support

    auto Q = node->input_value(0);
    auto K = node->input_value(1);
    auto V = node->input_value(2);
    auto past_key = node->input_value(3);
    auto past_value = node->input_value(4);
    auto seqlens_k = node->input_value(5);
    auto cos_cache = node->input_value(6);
    auto sin_cache = node->input_value(7);

    // The length of all tokens (past + current) is `seqlens_k` + 1.
    // current = Q.shape[2], past = `seqlens_k` + 1 - current

    const auto T = Q.get_element_type();
    const auto q_shape = register_new_node<v3::ShapeOf>(Q);
    const auto current_seqlen = get_dimensions(q_shape, {2});
    const auto head_size_node = get_dimensions(q_shape, {3});

    auto zero = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    auto zero_without_shape = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    auto one_without_shape = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto two = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    auto seqlens_elemi64 = register_new_node<v0::Convert>(seqlens_k, ov::element::i64);
    auto real_seqlens = register_new_node<v1::Add>(seqlens_elemi64, one);

    // Only consider batch is 1
    auto seqlens_1d = register_new_node<v1::Reshape>(real_seqlens, one, false);
    auto past_seqlen = register_new_node<v1::Subtract>(seqlens_1d, current_seqlen);
    auto curr_seqlen_scalar = register_new_node<v1::Reshape>(current_seqlen, one_without_shape, false);

    if (do_rotary) {
        Q = rotaryEmbedding(Q,
                            past_seqlen,
                            curr_seqlen_scalar,
                            cos_cache.get_node_shared_ptr(),
                            sin_cache.get_node_shared_ptr(),
                            head_size_node,
                            rotary_interleaved);
        K = rotaryEmbedding(K,
                            past_seqlen,
                            curr_seqlen_scalar,
                            cos_cache.get_node_shared_ptr(),
                            sin_cache.get_node_shared_ptr(),
                            head_size_node,
                            rotary_interleaved);
    }

    auto construct_kv_cache = [&](const ov::Output<ov::Node>& past, const ov::Output<ov::Node>& current) {
        return register_new_node<v0::Concat>(ov::OutputVector{past, current}, 2);
    };
    K = construct_kv_cache(past_key, K);
    V = construct_kv_cache(past_value, V);

    const auto concat_kv_len = get_dimensions(K.get_node_shared_ptr(), {2});
    const auto concat_kv_len_scalar = register_new_node<v1::Reshape>(concat_kv_len, one_without_shape, false);

    ov::Output<ov::Node> present_k;
    ov::Output<ov::Node> present_v;
    if (!is_static_input) {
        present_k = K;
        present_v = V;
    } else {
        const auto positions =
            register_new_node<v4::Range>(one_without_shape, concat_kv_len_scalar, one_without_shape, ov::element::i64);
        present_k = register_new_node<v8::Gather>(K, positions, two);
        present_v = register_new_node<v8::Gather>(V, positions, two);
    }

    // Broadcast KV if grouped query attention
    const size_t kv_num_heads_factor = num_heads / kv_num_heads;
    if (kv_num_heads_factor > 1) {
        const auto kv_shape = register_new_node<v3::ShapeOf>(K);
        const auto kv_shape_prev_2 = get_dimensions(kv_shape, {0, 1});
        const auto kv_shape_last_2 = get_dimensions(kv_shape, {2, 3});
        auto new_kv_shape = register_new_node<v0::Concat>(ov::NodeVector{kv_shape_prev_2, one, kv_shape_last_2}, 0);
        K = register_new_node<v1::Reshape>(K, new_kv_shape, false);
        V = register_new_node<v1::Reshape>(V, new_kv_shape, false);
        K = register_new_node<v0::Concat>(ov::OutputVector(kv_num_heads_factor, K), 2);
        V = register_new_node<v0::Concat>(ov::OutputVector(kv_num_heads_factor, V), 2);
        const auto q_shape = register_new_node<v3::ShapeOf>(Q);
        const auto q_shape_prev_2 = get_dimensions(q_shape, {0, 1});
        auto extended_kv_shape = register_new_node<v0::Concat>(ov::NodeVector{q_shape_prev_2, kv_shape_last_2}, 0);
        K = register_new_node<v1::Reshape>(K, extended_kv_shape, false);
        V = register_new_node<v1::Reshape>(V, extended_kv_shape, false);
    }

    // Make attention mask
    std::shared_ptr<ov::Node> mask;

    std::shared_ptr<ov::Node> hori_range =
        register_new_node<v4::Range>(zero_without_shape, concat_kv_len_scalar, one_without_shape, ov::element::i64);
    hori_range = register_new_node<v0::Unsqueeze>(hori_range, zero);

    std::shared_ptr<ov::Node> vert_range =
        register_new_node<v4::Range>(zero_without_shape, curr_seqlen_scalar, one_without_shape, ov::element::i64);
    vert_range = register_new_node<v0::Unsqueeze>(vert_range, one);
    const auto past_k_node_len = get_dimensions(past_key.get_node_shared_ptr(), {2});
    vert_range = register_new_node<v1::Add>(vert_range, past_k_node_len);

    const auto triu = register_new_node<v1::Greater>(hori_range, vert_range);
    const auto typed_zero = register_new_node(v0::Constant::create(T, ov::Shape{}, {0}));
    // cf. make_attention_mask@src\plugins\intel_gpu\tests\common\subgraphs_builders.hpp
    std::shared_ptr<ov::Node> minus_inf = nullptr;
    if (T == ov::element::f32)
        minus_inf = register_new_node(v0::Constant::create(T, ov::Shape{}, {-std::numeric_limits<float>::infinity()}));
    else if (T == ov::element::f16)
        minus_inf =
            register_new_node(v0::Constant::create(T, ov::Shape{}, {std::numeric_limits<ov::float16>::lowest()}));
    mask = register_new_node<v1::Select>(triu, minus_inf, typed_zero);

    if (is_static_input) {
        const auto padding_len = register_new_node<v1::Subtract>(concat_kv_len, seqlens_1d);
        const auto padding_mask_vert_shape = register_new_node<v0::Concat>(ov::NodeVector{current_seqlen, one}, 0);
        const auto padding_mask_vert = register_new_node<v3::Broadcast>(padding_len, padding_mask_vert_shape);
        const auto padding_mask = register_new_node<v1::GreaterEqual>(hori_range, padding_mask_vert);
        mask = register_new_node<v1::Select>(padding_mask, mask, minus_inf);
    }

    std::shared_ptr<ov::Node> qga_output;
    if (scale != 0.0f) {
        auto scale_node = register_new_node(v0::Constant::create(T, Shape{}, {scale}));
        qga_output = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, mask, scale_node, false);
    } else {
        qga_output = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, mask, false);
    }

    // transpose the result from (batch_size, num_heads, sequence_length, head_size)
    // to (batch_size, sequence_length, num_heads * head_size)
    auto perm = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3}));
    auto qga_output_transposed = register_new_node<v1::Transpose>(qga_output, perm);
    auto dim_merge_shape = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1}));
    auto output = register_new_node<v1::Reshape>(qga_output_transposed, dim_merge_shape, true)->output(0);

    return {output, present_k, present_v};
}

// make split functions is a copy-past from ONNX FE. TODO: move it to one place
ov::OutputVector ov::pass::GroupQueryAttentionDecomposition::make_split(const ov::Output<ov::Node>& value,
                                                                        int64_t num_splits,
                                                                        int64_t axis) {
    using namespace ov::op;
    const auto axis_node = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {axis}));
    const auto split = register_new_node<v1::Split>(value, axis_node, num_splits);

    return split->outputs();
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::get_dimensions(
    const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
    const std::vector<int>& dims) {
    using namespace ov::op;
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return register_new_node<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::get_dimensions(
    const std::shared_ptr<ov::Node>& node,
    const std::vector<int>& dims) {
    return get_dimensions(register_new_node<ov::op::v3::ShapeOf>(node), dims);
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::rotaryEmbedding(
    ov::Output<ov::Node> input,
    ov::Output<ov::Node> past_seqlen,
    ov::Output<ov::Node> curr_seqlen_scalar,
    std::shared_ptr<ov::Node> cos_cache,
    std::shared_ptr<ov::Node> sin_cache,
    std::shared_ptr<ov::Node> dim_head_size,
    bool interleaved) {
    using namespace ov::op;
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto zero_without_shape = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one_without_shape = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    ov::Output<ov::Node> position_ids =
        register_new_node<v4::Range>(zero_without_shape, curr_seqlen_scalar, one_without_shape, ov::element::i64);
    position_ids = register_new_node<v1::Add>(position_ids, past_seqlen);
    auto cos = register_new_node<v8::Gather>(cos_cache, position_ids, zero);
    auto sin = register_new_node<v8::Gather>(sin_cache, position_ids, zero);

    if (interleaved) {
        auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

        auto cache_shape = register_new_node<v3::ShapeOf>(cos_cache);
        auto cache_last_dim = get_dimensions(cos_cache, {-1});

        auto input_shape = register_new_node<v3::ShapeOf>(input);

        auto dim_bns = get_dimensions(input_shape, {0, 1, 2});
        std::shared_ptr<ov::Node> half_last_dim = cache_last_dim;

        auto negtive_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto split_input_shape = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, half_last_dim, two}, 0);
        auto reshaped_input = register_new_node<v1::Reshape>(input, split_input_shape, false);

        auto in_split = make_split(reshaped_input, 2, -1);
        split_input_shape = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, half_last_dim}, 0);
        auto in_split_0 = register_new_node<v1::Reshape>(in_split[0], split_input_shape, false);
        auto in_split_1 = register_new_node<v1::Reshape>(in_split[1], split_input_shape, false);

        auto res_0 = register_new_node<v1::Subtract>(register_new_node<v1::Multiply>(in_split_0, cos),
                                                     register_new_node<v1::Multiply>(in_split_1, sin));
        auto res_1 = register_new_node<v1::Add>(register_new_node<v1::Multiply>(in_split_0, sin),
                                                register_new_node<v1::Multiply>(in_split_1, cos));

        split_input_shape = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, half_last_dim, one}, 0);
        auto res_0_5d = register_new_node<v1::Reshape>(res_0, split_input_shape, false);
        auto res_1_5d = register_new_node<v1::Reshape>(res_1, split_input_shape, false);

        auto concat_ret = register_new_node<v0::Concat>(ov::NodeVector{res_0_5d, res_1_5d}, -1);
        return register_new_node<v1::Reshape>(concat_ret, input_shape, false);
    } else {
        auto in_split = make_split(input, 2, -1);
        auto res_0 = register_new_node<v1::Subtract>(register_new_node<v1::Multiply>(in_split[0], cos),
                                                     register_new_node<v1::Multiply>(in_split[1], sin));
        auto res_1 = register_new_node<v1::Add>(register_new_node<v1::Multiply>(in_split[0], sin),
                                                register_new_node<v1::Multiply>(in_split[1], cos));

        return register_new_node<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);
    }
}
