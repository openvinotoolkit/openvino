// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
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
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/convert_like.hpp"
#include "utils/split.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace detail {
namespace {

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
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<v3::ShapeOf>& shape, const std::vector<int>& dims);
}  // namespace
}  // namespace detail

namespace opset_1 {
ov::OutputVector group_query_attention(const ov::frontend::onnx::Node& node) {
    const auto do_rotary = node.get_attribute_value<int64_t>("do_rotary");
    const auto num_heads = node.get_attribute_value<int64_t>("num_heads");
    const auto kv_num_heads = node.get_attribute_value<int64_t>("kv_num_heads");
    const auto scale = node.get_attribute_value<float>("scale", 0.0f);
    const auto rotary_interleaved = node.get_attribute_value<float>("rotary_interleaved");
    // TODO: add softcap support

    auto nodes = node.get_ov_inputs();
    const auto node_shape = std::make_shared<v3::ShapeOf>(nodes[0]);
    const auto batch_size = detail::get_dimensions(node_shape, {0});
    const auto current_seqlen_size = detail::get_dimensions(node_shape, {1});
    const auto hidden_size = detail::get_dimensions(node_shape, {2});
    const auto total_num_heads_node =
        v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads + kv_num_heads + kv_num_heads});
    auto head_size_node = std::make_shared<v1::Divide>(hidden_size, total_num_heads_node);

    // transpose Q, K and V to (batch_size, num_heads, sequence_len, head_size)
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    // Q K V (batch_size, sequence_len, num_heads, head_size)
    ov::Output<ov::Node> Q, K, V;
    int index = 0;
    Q = nodes[index++];
    K = nodes[index++];
    V = nodes[index++];
    if (ov::op::util::is_null(K)) {
        // Handle the packed QKV
        auto packed_qkv_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size, current_seqlen_size, total_num_heads_node, head_size_node},
            0);
        auto inputs_qkv = std::make_shared<v1::Reshape>(Q, packed_qkv_shape, false)->output(0);
        // (batch_size, sequence_len, num_head, head_size)
        inputs_qkv = std::make_shared<v1::Transpose>(inputs_qkv, perm);
        // split the node into 3 even parts Q, K, V with shape (batch_size, num_head, sequence_len, head_size)
        auto split = ov::op::util::make_split(inputs_qkv, {num_heads, kv_num_heads, kv_num_heads}, 1);
        Q = split[0];
        K = split[1];
        V = split[2];
    } else {
        Q = std::make_shared<v1::Transpose>(Q, perm);
        K = std::make_shared<v1::Transpose>(K, perm);
        V = std::make_shared<v1::Transpose>(V, perm);
    }

    const auto& past_key = nodes[index++];
    const auto& past_value = nodes[index++];
    const auto& seqlens_k = nodes[index++].get_node_shared_ptr();
    const auto& total_sequence_length = nodes[index++];  // unused, it's not always equal (seqlens_k + 1)
    const auto& cos_cache = nodes[index++].get_node_shared_ptr();
    const auto& sin_cache = nodes[index++].get_node_shared_ptr();

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
        Q = detail::rotaryEmbedding(Q,
                                    past_sequence_length,
                                    seqlens_1d,
                                    cos_cache,
                                    sin_cache,
                                    head_size_node,
                                    rotary_interleaved);
        K = detail::rotaryEmbedding(K,
                                    past_sequence_length,
                                    seqlens_1d,
                                    cos_cache,
                                    sin_cache,
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
    auto present_k = K;
    auto present_v = V;

    std::shared_ptr<ov::Node> alpha;
    if (scale == 0.0f) {
        alpha = std::make_shared<v0::Sqrt>(head_size_node);
    } else {
        alpha = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f / scale});
    }
    const size_t kv_num_heads_factor = num_heads / kv_num_heads;
    if (kv_num_heads_factor > 1) {
        const auto kv_shape = std::make_shared<v3::ShapeOf>(K);
        // (batch_size, num_heads, sequence_len, head_size)
        const auto kv_shape_prev_2 = detail::get_dimensions(kv_shape, {0, 1});
        const auto kv_shape_last_2 = detail::get_dimensions(kv_shape, {2, 3});
        auto new_kv_shape = std::make_shared<v0::Concat>(ov::NodeVector{kv_shape_prev_2, one, kv_shape_last_2}, 0);
        K = std::make_shared<v1::Reshape>(K, new_kv_shape, false);
        V = std::make_shared<v1::Reshape>(V, new_kv_shape, false);
        K = std::make_shared<v0::Concat>(ov::OutputVector(kv_num_heads_factor, K), 2);
        V = std::make_shared<v0::Concat>(ov::OutputVector(kv_num_heads_factor, V), 2);
        auto q_shape = std::make_shared<v3::ShapeOf>(Q);
        // (batch_size, num_heads, sequence_len, head_size)
        const auto q_shape_prev_2 = detail::get_dimensions(q_shape, {0, 1});
        auto extended_kv_shape = std::make_shared<v0::Concat>(ov::NodeVector{q_shape_prev_2, kv_shape_last_2}, 0);
        K = std::make_shared<v1::Reshape>(K, extended_kv_shape, false);
        V = std::make_shared<v1::Reshape>(V, extended_kv_shape, false);
    }
    // compute softmax((Q x K') / sqrt(head_size))
    std::shared_ptr<ov::Node> softmax_input = std::make_shared<v0::MatMul>(Q, K, false, true);
    softmax_input = std::make_shared<v1::Divide>(softmax_input, alpha);

    // need to apply low-triangle mask to attention score.
    auto past_seq_len_scalar = std::make_shared<v1::Reshape>(past_sequence_length, one_without_shape, false);
    auto seqlens_1d_scalar = std::make_shared<v1::Reshape>(seqlens_1d, one_without_shape, false);
    std::shared_ptr<ov::Node> mask_per_line_node =
        std::make_shared<v4::Range>(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                    seqlens_1d_scalar,
                                    one_without_shape,
                                    ov::element::i64);
    mask_per_line_node = std::make_shared<v0::Unsqueeze>(mask_per_line_node, zero);
    auto minus_inf = v0::Constant::create(element::f32, Shape{}, {-std::numeric_limits<float>::infinity()});
    auto mask_shape = std::make_shared<v0::Concat>(ov::NodeVector{current_seqlen_size, seqlens_1d}, 0);
    auto compare_mask = std::make_shared<v3::Broadcast>(mask_per_line_node, mask_shape);

    std::shared_ptr<ov::Node> vertical_range =
        std::make_shared<v4::Range>(past_seq_len_scalar, seqlens_1d_scalar, one_without_shape, ov::element::i64);
    vertical_range = std::make_shared<v0::Unsqueeze>(vertical_range, one);

    auto triu = std::make_shared<v1::Greater>(compare_mask, vertical_range);
    auto typed_zero = std::make_shared<v1::ConvertLike>(zero, softmax_input);
    auto typed_minus_inf = std::make_shared<v1::ConvertLike>(minus_inf, softmax_input);
    auto minus_inf_mask = std::make_shared<v3::Broadcast>(typed_minus_inf, mask_shape);
    auto atten_mask = std::make_shared<v1::Select>(triu, minus_inf_mask, typed_zero);

    std::shared_ptr<ov::Node> softmax_input_added = std::make_shared<v1::Add>(softmax_input, atten_mask);
    // softmax((Q x K' + mask) / sqrt(head_size))
    const auto softmax = std::make_shared<v8::Softmax>(softmax_input_added, 3);

    // softmax((Q x K' + mask) / sqrt(head_size)) x V
    std::shared_ptr<ov::Node> output = std::make_shared<v0::MatMul>(softmax, V);

    // transpose the result from (batch_size, num_heads, sequence_length, head_size)
    // to (batch_size, sequence_length, num_heads, head_size)
    output = std::make_shared<v1::Transpose>(output, perm);
    auto dim_merge_shape = v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1});
    // reshape the result from (batch_size, sequence_length, num_heads, head_size)
    //          to             (batch_size, sequence_length, num_heads * head_size)
    output = std::make_shared<v1::Reshape>(output, dim_merge_shape, true);

    return {output, present_k, present_v};
}

ONNX_OP("GroupQueryAttention", OPSET_SINCE(1), com_microsoft::opset_1::group_query_attention, MICROSOFT_DOMAIN);

}  // namespace opset_1

namespace detail {
namespace {

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<v3::ShapeOf>& shape, const std::vector<int>& dims) {
    static const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims) {
    return get_dimensions(std::make_shared<v3::ShapeOf>(node), dims);
}

std::shared_ptr<ov::Node> rotaryEmbedding(ov::Output<ov::Node> input,
                                          ov::Output<ov::Node> past_seqlen,
                                          std::shared_ptr<ov::Node> seqlen_k,
                                          std::shared_ptr<ov::Node> cos_cache,
                                          std::shared_ptr<ov::Node> sin_cache,
                                          std::shared_ptr<ov::Node> dim_head_size,
                                          bool interleaved) {
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    auto cache_shape = std::make_shared<v3::ShapeOf>(cos_cache);
    auto cache_last_dim = get_dimensions(cache_shape, {-1});
    auto cache_1st_dim = get_dimensions(cache_shape, {0});

    // TODO: check the shape
    auto input_shape = std::make_shared<v3::ShapeOf>(input);

    auto dim_bns = get_dimensions(input_shape, {0, 1, 2});
    // auto dim_head_size = get_dimensions(input_shape, {3});
    // half_last_dim is same as cos_cache
    std::shared_ptr<ov::Node> half_last_dim = cache_last_dim;

    auto real_cache_shape = std::make_shared<v0::Concat>(ov::NodeVector{cache_1st_dim, dim_head_size}, 0);
    auto slice_cache_dim_shape = seqlen_k;

    // auto end_lens = std::make_shared<v1::Subtract>(half_last_dim, one);
    // auto masks = std::make_shared<v12::Pad>(one,
    //                                         zero,
    //                                         end_lens,
    //                                         op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}),
    //                                         ov::op::PadMode::CONSTANT);
    auto masks = std::make_shared<v3::Broadcast>(one, half_last_dim);

    if (interleaved) {
        auto negtive_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto split_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, half_last_dim, two}, 0);
        auto reshaped_input = std::make_shared<v1::Reshape>(input, split_input_shape, false);
        auto first_half = std::make_shared<v8::Slice>(reshaped_input, zero, one, one, negtive_one);
        auto second_half = std::make_shared<v8::Slice>(reshaped_input, one, two, one, negtive_one);

        auto second_input = std::make_shared<v0::Concat>(ov::NodeVector{second_half, first_half}, -1);

        auto mask_shape = std::make_shared<v0::Concat>(ov::NodeVector{half_last_dim, one}, 0);
        auto reshaped_mask = std::make_shared<v1::Reshape>(masks, mask_shape, false);
        auto negtive_mask = std::make_shared<v0::Negative>(reshaped_mask);
        auto concat_mask = std::make_shared<v0::Concat>(ov::NodeVector{negtive_mask, reshaped_mask}, -1);
        auto real_mask = std::make_shared<v1::Reshape>(concat_mask, dim_head_size, false);
        auto mask_f32 = std::make_shared<v0::Convert>(real_mask, ov::element::f32);

        auto real_input0 = std::make_shared<v1::Reshape>(reshaped_input, input_shape, false);
        auto real_input1 = std::make_shared<v1::Reshape>(second_input, input_shape, false);

        auto new_cache_shape = std::make_shared<v0::Concat>(ov::NodeVector{cache_shape, two}, 0);
        auto temp_cache_shape = std::make_shared<v0::Concat>(ov::NodeVector{cache_shape, one}, 0);
        auto cos_cache_reshape = std::make_shared<v1::Reshape>(cos_cache, temp_cache_shape, false);
        auto sin_cache_reshape = std::make_shared<v1::Reshape>(sin_cache, temp_cache_shape, false);
        auto cos_cache_broadcasted = std::make_shared<v3::Broadcast>(cos_cache_reshape, new_cache_shape);
        auto sin_cache_broadcasted = std::make_shared<v3::Broadcast>(sin_cache_reshape, new_cache_shape);
        auto real_cos_input = std::make_shared<v1::Reshape>(cos_cache_broadcasted, real_cache_shape, false);
        auto real_sin_input = std::make_shared<v1::Reshape>(sin_cache_broadcasted, real_cache_shape, false);
        auto sliced_cos_input =
            std::make_shared<v8::Slice>(real_cos_input, past_seqlen, slice_cache_dim_shape, one, zero);
        auto sliced_sin_input =
            std::make_shared<v8::Slice>(real_sin_input, past_seqlen, slice_cache_dim_shape, one, zero);
        auto add_input0 = std::make_shared<v1::Multiply>(real_input0, sliced_cos_input);
        auto add_input1 = std::make_shared<v1::Multiply>(real_input1, sliced_sin_input);
        auto multi_input1 = std::make_shared<v1::Multiply>(add_input1, mask_f32);
        auto result = std::make_shared<v1::Add>(add_input0, multi_input1);
        return result;
    } else {
        auto cos =
            std::make_shared<v8::Slice>(cos_cache, past_seqlen, slice_cache_dim_shape, one, zero);
        auto sin =
            std::make_shared<v8::Slice>(sin_cache, past_seqlen, slice_cache_dim_shape, one, zero);
        auto in_split = ov::op::util::make_split(input, 2, -1);
        auto res_0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(in_split[0], cos), std::make_shared<v1::Multiply>(in_split[1], sin));
        auto res_1 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(in_split[0], sin), std::make_shared<v1::Multiply>(in_split[1], cos));

        return std::make_shared<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);
    }
}
}  // namespace
}  // namespace detail
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
