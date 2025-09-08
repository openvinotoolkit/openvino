// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"

#include <algorithm>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "utils/common.hpp"
#include "utils/split.hpp"

using namespace ov::op;

namespace ov::frontend::onnx::com_microsoft {

namespace detail {
namespace {
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<v3::ShapeOf>& shape, const std::vector<int>& dims);
}  // namespace
}  // namespace detail

namespace opset_1 {
ov::OutputVector group_query_attention(const ov::frontend::onnx::Node& node) {
    // At least given "query" and "seqlens_k"
    common::default_op_checks(node, 2);

    const auto onnx_op_inputs = node.get_ov_inputs();
    const auto num_heads = node.get_attribute_value<int64_t>("num_heads");
    const auto kv_num_heads = node.get_attribute_value<int64_t>("kv_num_heads");
    const auto scale = node.get_attribute_value<float>("scale", 0.0f);
    const auto do_rotary = node.get_attribute_value<int64_t>("do_rotary", 0);
    const auto rotary_interleaved = node.get_attribute_value<int64_t>("rotary_interleaved", 0);

    // In ONNX, the format of input QKV is [B, S, N*H] and of past_kv is [B, N, S, H]
    // In OV, we always use [B, N, S, H]
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    auto Q = onnx_op_inputs[0];
    auto K = onnx_op_inputs[1];
    auto V = onnx_op_inputs[2];
    const auto q_shape_node = std::make_shared<v3::ShapeOf>(Q);
    const auto batch_size_node = detail::get_dimensions(q_shape_node, {0});
    const auto current_seqlen_size_node = detail::get_dimensions(q_shape_node, {1});
    const auto hidden_size_node = detail::get_dimensions(q_shape_node, {2});

    OutputVector ov_op_inputs;
    if (ov::op::util::is_null(K)) {
        auto total_num_heads_node =
            v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads + kv_num_heads + kv_num_heads});
        auto head_size_node = std::make_shared<v1::Divide>(hidden_size_node, total_num_heads_node);
        auto packed_qkv_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size_node, current_seqlen_size_node, total_num_heads_node, head_size_node},
            0);

        auto inputs_qkv = std::make_shared<v1::Reshape>(Q, packed_qkv_shape, false)->output(0);
        inputs_qkv = std::make_shared<v1::Transpose>(inputs_qkv, perm);
        auto split = ov::op::util::make_split(inputs_qkv, {num_heads, kv_num_heads, kv_num_heads}, 1);

        std::copy(split.begin(), split.end(), std::back_inserter(ov_op_inputs));
    } else {
        auto num_heads_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads});
        auto head_size_node = std::make_shared<v1::Divide>(hidden_size_node, num_heads_node);
        auto q_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size_node, current_seqlen_size_node, num_heads_node, head_size_node},
            0);

        Q = std::make_shared<v1::Reshape>(Q, q_shape, false)->output(0);
        Q = std::make_shared<v1::Transpose>(Q, perm);
        ov_op_inputs.push_back(Q);

        auto kv_num_heads_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {kv_num_heads});
        auto kv_shape = std::make_shared<v0::Concat>(
            ov::NodeVector{batch_size_node, current_seqlen_size_node, kv_num_heads_node, head_size_node},
            0);

        K = std::make_shared<v1::Reshape>(K, kv_shape, false)->output(0);
        V = std::make_shared<v1::Reshape>(V, kv_shape, false)->output(0);
        K = std::make_shared<v1::Transpose>(K, perm);
        V = std::make_shared<v1::Transpose>(V, perm);
        ov_op_inputs.push_back(K);
        ov_op_inputs.push_back(V);
    }

    for (int i = 3; i < 9; ++i) {
        // skip total_sequence_length
        if (i == 6)
            continue;
        ov_op_inputs.push_back(onnx_op_inputs[i]);
    }
    return std::make_shared<internal::GroupQueryAttention>(ov_op_inputs,
                                                           num_heads,
                                                           kv_num_heads,
                                                           scale,
                                                           do_rotary,
                                                           rotary_interleaved)
        ->outputs();
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
}  // namespace
}  // namespace detail

}  // namespace ov::frontend::onnx::com_microsoft
