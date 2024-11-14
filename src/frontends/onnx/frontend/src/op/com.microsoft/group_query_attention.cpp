// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"

// TODO: Filter out unused headers

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
#include "openvino/op/log.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

#include "utils/split.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace detail {
namespace {

// FIXME: Reuse the same function from file attention.cpp, but it requires a bit of adaptation -- I have redesigned part of the inputs a bit here and in the helper functions below
ov::NodeVector split_to_QKV(const Output<ov::Node>& node,
                            int64_t num_heads,
                            const std::vector<int64_t>& qkv_hidden_sizes);

ov::Output<ov::Node> get_elements(const ov::Output<ov::Node>& shape, const std::vector<int>& dims) {
    static const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

ov::Output<ov::Node> get_dimensions(const ov::Output<ov::Node>& node, const std::vector<int>& dims) {
    return get_elements(std::make_shared<v3::ShapeOf>(node), dims);
}

ov::Output<ov::Node> rope(
    const Output<ov::Node>& x,
    Output<ov::Node> cos,
    Output<ov::Node> sin,
    bool interleaved,
    const Output<ov::Node>& head_size,
    const Output<ov::Node>& pos_id_begin,
    const Output<ov::Node>& pos_id_end
) {
    OPENVINO_ASSERT(!interleaved, "rotary_interleaved is not supported");  // TODO: Support interleaved mode

    using v1::Split;
    using v0::Constant;
    using v1::Multiply;
    using v1::Add;
    using v8::Slice;
    using v0::Concat;
    using v1::Subtract;
    using Output = Output<ov::Node>;
    using std::make_shared;

    Output zero = Constant::create(element::i32, Shape{1}, {0});
    Output step = Constant::create(element::i32, Shape{1}, {1});

    // cut for the current sequence length
    cos = make_shared<Slice>(cos, pos_id_begin, pos_id_end, step, zero);
    sin = make_shared<Slice>(sin, pos_id_begin, pos_id_end, step, zero);

    OutputVector x_split = make_shared<Split>(x, Constant::create(element::i32, Shape{}, {-1}), 2)->outputs();

    Output res_0 = make_shared<Subtract>(
        make_shared<Multiply>(x_split[0], cos),
        make_shared<Multiply>(x_split[1], sin)
    );

    Output res_1 = make_shared<Add>(
        make_shared<Multiply>(x_split[0], sin),
        make_shared<Multiply>(x_split[1], cos)
    );

    return make_shared<Concat>(OutputVector{res_0, res_1}, -1);
}

ov::Output<ov::Node> broadcast_groups(const Output<ov::Node>& cache, const int num_kv_heads, const int num_heads) {
    if(num_kv_heads == 1 || num_kv_heads == num_heads) {
        // No broadcast or there is the broadcast that SDPA broadcastability can handle
        return cache;
    }

    OPENVINO_ASSERT(num_heads % num_kv_heads == 0);
    const auto broadcast_multiplier = num_heads/num_kv_heads;

    auto unsqueeze = std::make_shared<v0::Unsqueeze>(cache, v0::Constant::create(element::i32, Shape{}, {2}));
    auto shapeof = std::make_shared<v3::ShapeOf>(cache, element::i32);

    auto broadcast_shape = std::make_shared<v0::Concat>(OutputVector{
        get_elements(shapeof, {0, 1}),
        v0::Constant::create(element::i32, Shape{1}, {broadcast_multiplier}),
        get_elements(shapeof, {2, 3})
    }, 0);

    auto broadcast = std::make_shared<v3::Broadcast>(unsqueeze, broadcast_shape);

    auto reshape_shape = std::make_shared<v0::Concat>(OutputVector{
        v0::Constant::create(element::i32, Shape{3}, {0, num_heads, -1}),
        get_elements(shapeof, {3})
    }, 0);

    auto reshape = std::make_shared<v1::Reshape>(broadcast, reshape_shape, true);

    return reshape;
}


}  // namespace
}  // namespace detail

namespace opset_1 {
ov::OutputVector group_query_attention(const ov::frontend::onnx::Node& node) {
    auto nodes = node.get_ov_inputs();
    const auto& input = nodes[0];

    ov::Output<ov::Node> Q, K, V, head_size;

    const auto num_heads = node.get_attribute_value<int64_t>("num_heads");

    if(ov::op::util::is_null(nodes[1]) || ov::op::util::is_null(nodes[2])) {
        const auto split_result = detail::split_to_QKV(input, static_cast<int>(num_heads), {});
        Q = split_result[0];
        K = split_result[1];
        V = split_result[2];
        head_size = split_result[3];
    } else {
        Q = input;
        K = nodes[1];
        V = nodes[2];
        head_size = detail::get_dimensions(Q, {-1});
    }

    const auto& past_K = nodes[3];
    const auto& past_V = nodes[4];
    const auto& seqlens_k = nodes[5];
    const auto& total_sequence_length = nodes[6];
    const auto& cos = nodes[7];
    const auto& sin = nodes[8];
    const bool rope_interleaved = node.get_attribute_value<int64_t>("rotary_interleaved", 0);

    // FIXME: It works only when KV cache is dynamically growing and doesn't have unused space inside. So it is not compatible with statically-shaped KV cache.
    // const auto past_seq_len = detail::get_dimensions(past_K, {0});
    // TODO: GQA spec is not compatible with test model. Spec supposes 1D tensor, in the test model we have 2D tensor, flattening to work in both cases.

    // FIXME: Unaligned elements in KV cache are not supported.
    // We just get one of the seq lens as a common value for all past sequences
    const auto& past_seq_len = detail::get_elements(std::make_shared<v1::Reshape>(seqlens_k, v0::Constant::create(element::i32, Shape{1}, {-1}), false), {0});

    Q = detail::rope(Q, cos, sin, rope_interleaved, head_size, past_seq_len, total_sequence_length);
    K = detail::rope(K, cos, sin, rope_interleaved, head_size, past_seq_len, total_sequence_length);

    K = std::make_shared<v0::Concat>(ov::OutputVector{past_K, K}, 2);
    V = std::make_shared<v0::Concat>(ov::OutputVector{past_V, V}, 2);

    const auto num_kv_heads = node.get_attribute_value<int64_t>("kv_num_heads");

    K = detail::broadcast_groups(K, num_kv_heads, num_heads);
    V = detail::broadcast_groups(V, num_kv_heads, num_heads);

    // FIXME: Unaligned batch of sequences is not supported. All past key-value are assumed to have the same length.
    // That means all input sequence lengths should be the same and match input.shape[2]
    // We do not check that here because it depends on runtime values.
    // If we want to implement not aligned batch of dimensions we have to form not uniform causal mask for attention that
    // adds a significant porition of the code.

    // FIXME: The same tensor at input/output of past/preset K and V are not supported.
    // It requires more complex tensor manipulations that are introduce overhead into pure tensor-value data flow and should be implemented if we really have demand for that.
    // Also inplace KV-cache modification logic is not supported efficiently in any plugins (CPU, GPU and NPU).

    auto output = std::make_shared<v13::ScaledDotProductAttention>(Q, K, V, true);

    return {output, K, V};
}
ONNX_OP("GroupQueryAttention", OPSET_SINCE(1), com_microsoft::opset_1::group_query_attention, MICROSOFT_DOMAIN);
}  // namespace opset_1

namespace detail {
namespace {



std::shared_ptr<ov::Node> get_hidden_size(const std::shared_ptr<v3::ShapeOf>& node_shape) {
    // node has shape (batch_size, sequence_length, 3 * hidden_size)
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto hidden_size_x3 = get_elements(node_shape, {2});
    const auto three = v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
    const auto hidden_size = std::make_shared<v1::Divide>(hidden_size_x3, three);
    return hidden_size;
}

ov::NodeVector split_to_QKV(const Output<ov::Node>& node,
                            int64_t num_heads,
                            const std::vector<int64_t>& qkv_hidden_sizes) {
    ov::OutputVector split;
    std::shared_ptr<ov::Node> head_size = nullptr;
    const auto& node_type = node.get_element_type();
    const auto node_shape = std::make_shared<v3::ShapeOf>(node);
    // node has shape (batch_size, sequence_length, 3 * hidden_size)
    // fetch the first two dimensions
    const auto batch_size_seq_len = get_elements(node_shape, {0, 1});
    const auto num_heads_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads});
    if (qkv_hidden_sizes.size() == 0) {
        const auto hidden_size = get_hidden_size(node_shape);
        // head_size = hidden_size / num_heads
        head_size = std::make_shared<v1::Divide>(hidden_size, num_heads_node);
        // split the node into 3 even parts Q, K, V with shape (batch_size, sequence_len, hidden_size)
        split = ov::op::util::make_split(node, 3, 2);
        // and reshape each part to new shape (batch_size, sequence_len, num_heads, head_size)
        auto new_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch_size_seq_len, num_heads_node, head_size}, 0);
        for (size_t i = 0; i < split.size(); i++) {
            split[i] = std::make_shared<v1::Reshape>(split[i], new_shape, false);
        }
        head_size = std::make_shared<v0::Convert>(head_size, node_type);
    } else {
        // in this case, weights have shape
        // (input_hidden_size, qkv_hidden_sizes[0] + qkv_hidden_sizes[1] + qkv_hidden_sizes[2])
        // so user specified hidden_sizes for Q, K and V
        FRONT_END_GENERAL_CHECK(qkv_hidden_sizes.size() == 3, "qkv_hidden_sizes attribute needs to have 3 values");
        FRONT_END_GENERAL_CHECK(qkv_hidden_sizes[0] == qkv_hidden_sizes[1],
                                "qkv_hidden_sizes first element should be same as the second");
        // split the node into 3 parts Q, K, V with shapes
        // Q: (batch_size, sequence_len, qkv_hidden_sizes[0])
        // K: (batch_size, sequence_len, qkv_hidden_sizes[1])
        // V: (batch_size, sequence_len, qkv_hidden_sizes[2])
        split = ov::op::util::make_split(node, qkv_hidden_sizes, 2);
        // and reshape each part to new shape (batch_size, sequence_len, num_heads, head_size)
        for (size_t i = 0; i < split.size(); i++) {
            auto new_shape = std::make_shared<v0::Concat>(
                ov::OutputVector{batch_size_seq_len,
                               num_heads_node,
                               v0::Constant::create(ov::element::i64, ov::Shape{1}, {qkv_hidden_sizes[i] / num_heads})},
                0);
            split[i] = std::make_shared<v1::Reshape>(split[i], new_shape, false);
        }
        float head_size_val = qkv_hidden_sizes[0] > 0 ? static_cast<float>(qkv_hidden_sizes[0]) / num_heads
                                                      : static_cast<float>(qkv_hidden_sizes[2]) / num_heads;
        head_size = v0::Constant::create(node_type, ov::Shape{1}, {head_size_val});
    }

    // transpose Q, K and V to (batch_size, num_heads, sequence_len, head_size)
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    auto Q = std::make_shared<v1::Transpose>(split[0], perm);
    auto K = std::make_shared<v1::Transpose>(split[1], perm);
    auto V = std::make_shared<v1::Transpose>(split[2], perm);

    return {Q, K, V, head_size};
}


}  // namespace
}  // namespace detail
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
