// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
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
    // TODO: add softcap support

    auto Q = node->input_value(0);
    auto K = node->input_value(1);
    auto V = node->input_value(2);
    auto past_key = node->input_value(3);
    auto past_value = node->input_value(4);
    auto seqlens_k = node->input_value(5);
    auto cos_cache = node->input_value(6);
    auto sin_cache = node->input_value(7);

    // The length of all tokens (past + current) is `seqlens_k` + 1
    // current = Q.shape[2], past = `seqlens_k` + 1 - current

    const auto T = Q.get_element_type();
    const auto q_shape = register_new_node<v3::ShapeOf>(Q);
    const auto current_sequence_length = get_dimensions(q_shape, {2});
    const auto head_size_node = get_dimensions(q_shape, {3});

    auto zero = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    auto one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    auto one_without_shape = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    auto two = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    auto seqlens_elemi64 = register_new_node<v0::Convert>(seqlens_k, ov::element::i64);
    auto real_seqlens = register_new_node<v1::Add>(seqlens_elemi64, one);

    // Only consider batch is 1
    auto seqlens_1d = register_new_node<v1::Reshape>(real_seqlens, one, false);
    auto past_sequence_length = register_new_node<v1::Subtract>(seqlens_1d, current_sequence_length);
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

    auto construct_kv_cache = [&](const ov::Output<ov::Node>& past, const ov::Output<ov::Node>& current) {
        auto past_datas = register_new_node<v8::Slice>(past, zero, past_sequence_length, one, two);
        auto curr_datas = register_new_node<v8::Slice>(current, zero, current_sequence_length, one, two);
        return register_new_node<v0::Concat>(ov::NodeVector{past_datas, curr_datas}, 2);
    };
    K = construct_kv_cache(past_key, K);
    V = construct_kv_cache(past_value, V);
    auto present_k = K;
    auto present_v = V;

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

    // need to apply low-triangle mask to attention score.
    // two steps, construct the total_sequence x total_sequence triangle, then slice the current length
    auto seqlens_1d_scalar = register_new_node<v1::Reshape>(seqlens_1d, one_without_shape, false);
    std::shared_ptr<ov::Node> mask_per_line_node =
        register_new_node<v4::Range>(register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {0})),
                                     seqlens_1d_scalar,
                                     one_without_shape,
                                     ov::element::i64);
    auto hori_range = register_new_node<v0::Unsqueeze>(mask_per_line_node, zero);
    auto vert_range = register_new_node<v0::Unsqueeze>(mask_per_line_node, one);
    auto triu = register_new_node<v1::Greater>(hori_range, vert_range);
    auto typed_zero = register_new_node(v0::Constant::create(T, ov::Shape{}, {0}));
    // cf. make_attention_mask@src\plugins\intel_gpu\tests\common\subgraphs_builders.hpp
    std::shared_ptr<ov::Node> minus_inf = nullptr;
    if (T == ov::element::f32)
        minus_inf = register_new_node(v0::Constant::create(T, ov::Shape{}, {-std::numeric_limits<float>::infinity()}));
    else if (T == ov::element::f16)
        minus_inf =
            register_new_node(v0::Constant::create(T, ov::Shape{}, {std::numeric_limits<ov::float16>::lowest()}));
    auto atten_mask = register_new_node<v1::Select>(triu, minus_inf, typed_zero);
    auto atten_mask_sliced = register_new_node<v8::Slice>(atten_mask, past_sequence_length, seqlens_1d, one, zero);

    std::shared_ptr<ov::Node> qga_output;
    if (scale != 0.0f) {
        auto scale_node = register_new_node(v0::Constant::create(T, Shape{}, {scale}));
        qga_output = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, atten_mask_sliced, scale_node, false);
    } else {
        qga_output = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, atten_mask_sliced, false);
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
    std::shared_ptr<ov::Node> seqlen_k,
    std::shared_ptr<ov::Node> cos_cache,
    std::shared_ptr<ov::Node> sin_cache,
    std::shared_ptr<ov::Node> dim_head_size,
    bool interleaved) {
    using namespace ov::op;
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    auto slice_cache_dim_shape = seqlen_k;

    auto cos = register_new_node<v8::Slice>(cos_cache, past_seqlen, slice_cache_dim_shape, one, zero);
    auto sin = register_new_node<v8::Slice>(sin_cache, past_seqlen, slice_cache_dim_shape, one, zero);

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
