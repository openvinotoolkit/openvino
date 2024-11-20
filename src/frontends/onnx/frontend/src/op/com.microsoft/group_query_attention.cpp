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
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/scatter_elements_update.hpp"
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


using v1::Split;
using v0::Constant;
using v1::Multiply;
using v1::Add;
using v8::Slice;
using v0::Concat;
using v1::Subtract;
using v3::ShapeOf;
using v3::Broadcast;
using v1::Reshape;
using v0::Unsqueeze;
using v4::Range;
using v3::ScatterUpdate;
using v3::ScatterElementsUpdate;
using v15::Squeeze;
using Output = ov::Output<ov::Node>;
using std::make_shared;

// FIXME: Reuse the same function from file attention.cpp, but it requires a bit of adaptation -- I have redesigned part of the inputs a bit here and in the helper functions below
ov::NodeVector split_to_QKV(const Output& node,
                            int64_t num_heads,
                            const std::vector<int64_t>& qkv_hidden_sizes);

Output get_elements(const Output& shape, const std::vector<int>& dims) {
    static const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

Output get_dimensions(const Output& node, const std::vector<int>& dims) {
    return get_elements(std::make_shared<v3::ShapeOf>(node, element::i32), dims);
}

std::shared_ptr<ov::Node> attention_softmax(Output& Q,
                                            Output& K,
                                            Output& V,
                                            Output& head_size);

Output rope(
    const Output& x,
    const Output& cos_cache,
    const Output& sin_cache,
    bool interleaved,
    const Output& head_size,
    const Output& pos_id_begin,
    const Output& pos_id_end
) {
    OPENVINO_ASSERT(!interleaved, "rotary_interleaved is not supported");  // TODO: Support interleaved mode

    Output zero = Constant::create(element::i32, Shape{1}, {0});
    Output step = Constant::create(element::i32, Shape{1}, {1});

    // cut for the current sequence length
    Output cos = make_shared<Slice>(cos_cache, pos_id_begin, pos_id_end, step, zero);
    Output sin = make_shared<Slice>(sin_cache, pos_id_begin, pos_id_end, step, zero);

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

Output broadcast_groups(const Output& cache, const int num_kv_heads, const int num_heads) {
    if(num_kv_heads == 1 || num_kv_heads == num_heads) {
        // No broadcast or there is the broadcast that SDPA broadcastability can handle
        return cache;
    }

    OPENVINO_ASSERT(num_heads % num_kv_heads == 0);
    const auto broadcast_multiplier = num_heads/num_kv_heads;

    auto unsqueeze = make_shared<Unsqueeze>(cache, Constant::create(element::i32, Shape{}, {2}));
    auto shapeof = make_shared<ShapeOf>(cache, element::i32);

    auto broadcast_shape = make_shared<Concat>(OutputVector{
        get_elements(shapeof, {0, 1}),
        Constant::create(element::i32, Shape{1}, {broadcast_multiplier}),
        get_elements(shapeof, {2, 3})
    }, 0);

    auto broadcast = make_shared<Broadcast>(unsqueeze, broadcast_shape);

    auto reshape_shape = make_shared<Concat>(OutputVector{
        Constant::create(element::i32, Shape{3}, {0, num_heads, -1}),
        get_elements(shapeof, {3})
    }, 0);

    auto reshape = make_shared<Reshape>(broadcast, reshape_shape, true);

    return reshape;
}

Output concat_cache(const Output& past, const Output& current) {
    return make_shared<Concat>(ov::OutputVector{past, current}, 2);     // 2 is the dimension index that corresponds to sequence len
}

Output squeeze_1d(const Output& x) {
    return make_shared<Squeeze>(x, Constant::create(element::i32, Shape{0}, {1}));
}

Output update_cache(const Output& past, const Output& current, const Output& past_len) {
    Output update_len = get_dimensions(current, {2});
    const auto one = Constant::create(element::i32, Shape{}, {1});
    Output update_end = make_shared<Add>(past_len, update_len);
    Output update_indices = make_shared<Range>(squeeze_1d(past_len), squeeze_1d(update_end), one, element::i32);
    auto shape_of = make_shared<ShapeOf>(current, element::i32);
    auto unsqueezed_indices = make_shared<Unsqueeze>(update_indices, Constant::create(element::i32, Shape{3}, {0,1,3}));
    auto broadcast_indices = make_shared<Broadcast>(unsqueezed_indices, shape_of);
    return make_shared<ScatterElementsUpdate>(past, broadcast_indices, current, Constant::create(element::i32, Shape{1}, {2}));
    // ScatterUpdate correct in CPU but fail in GPU
    // return make_shared<ScatterUpdate>(past, update_indices, current, Constant::create(element::i32, Shape{1}, {2}));    // 2 is the dimension index that corresponds to sequence len
}

ov::OutputVector group_query_attention_decomposition(
    const ov::OutputVector& inputs,
    int num_heads,
    bool rotary_interleaved,
    int kv_num_heads
) {
    const auto& input = inputs[0];

    Output Q, K, V, head_size;

    if(ov::op::util::is_null(inputs[1]) || ov::op::util::is_null(inputs[2])) {
        // split by num_head, kv_num_heads, kv_num_heads in N dim
        int64_t qkv_hidden_size =  input.get_partial_shape().get_shape()[2];
        int64_t per_head_size = qkv_hidden_size / (num_heads + kv_num_heads*2);
        const std::vector<int64_t> qkv_sizes = {num_heads*per_head_size, kv_num_heads*per_head_size, kv_num_heads*per_head_size};
        const auto split_result = detail::split_to_QKV(input, num_heads, qkv_sizes);
        Q = split_result[0];
        K = split_result[1];
        V = split_result[2];
        head_size = split_result[3];
    } else {
        Q = input;
        K = inputs[1];
        V = inputs[2];
        head_size = detail::get_dimensions(Q, {-1});
    }
    
    const auto q_shape = get_dimensions(Q, {0, 1, 2, 3});
    const auto k_shape = get_dimensions(K, {0, 1, 2, 3});
    const auto& past_K = inputs[3];
    const auto& past_V = inputs[4];
    const auto& seqlens_k = inputs[5];
    const auto& total_sequence_length = inputs[6];
    const auto& cos = inputs[7];
    const auto& sin = inputs[8];

    // FIXME: It works only when KV cache is dynamically growing and doesn't have unused space inside. So it is not compatible with statically-shaped KV cache.
    // const auto past_seq_len = detail::get_dimensions(past_K, {0});
    // TODO: GQA spec is not compatible with test model. Spec supposes 1D tensor, in the test model we have 2D tensor, flattening to work in both cases.

    // FIXME: Unaligned elements in KV cache are not supported.
    // We just get the first of the seq lens as a common value for all past sequences ignoring others, under assumption that they are all the same
    const auto& past_seq_len = detail::get_elements(std::make_shared<v1::Reshape>(seqlens_k, v0::Constant::create(element::i32, Shape{1}, {-1}), false), {0});

    Q = rope(Q, cos, sin, rotary_interleaved, head_size, past_seq_len, total_sequence_length);
    Q = make_shared<v1::Reshape>(Q, q_shape, true);
    K = rope(K, cos, sin, rotary_interleaved, head_size, past_seq_len, total_sequence_length);
    K = make_shared<v1::Reshape>(K, k_shape, true);

    if(past_K.get_partial_shape()[2].is_dynamic()) {
        K = concat_cache(past_K, K);
    } else {
        K = update_cache(past_K, K, past_seq_len);
    }

    if(past_V.get_partial_shape()[2].is_dynamic()) {
        V = concat_cache(past_V, V);
    } else {
        V = update_cache(past_V, V, past_seq_len);
    }

    K = broadcast_groups(K, kv_num_heads, num_heads);
    V = broadcast_groups(V, kv_num_heads, num_heads);

    Output zero = Constant::create(element::i32, Shape{1}, {0});
    Output one = Constant::create(element::i32, Shape{1}, {1});
    Output two = Constant::create(element::i32, Shape{1}, {2});
    Output K_compute = make_shared<Slice>(K, zero, total_sequence_length, one, two);
    Output V_compute = make_shared<Slice>(V, zero, total_sequence_length, one, two);
    // FIXME: Unaligned batch of sequences is not supported. All past key-value are assumed to have the same length.
    // That means all input sequence lengths should be the same and match input.shape[2]
    // We do not check that here because it depends on runtime values.
    // If we want to implement not aligned batch of dimensions we have to form not uniform causal mask for attention that
    // adds a significant porition of the code.

    // FIXME: The same tensor at input/output of past/preset K and V are not supported.
    // It requires more complex tensor manipulations that are introduce overhead into pure tensor-value data flow and should be implemented if we really have demand for that.
    // Also inplace KV-cache modification logic is not supported efficiently in any plugins (CPU, GPU and NPU).

    // TODO: check SDPA fail if past_len>0
    // Output output = make_shared<v13::ScaledDotProductAttention>(Q, K_compute, V_compute, true);
    // // permute & reorder to BSH format
    // const auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    // output = make_shared<v1::Transpose>(output, perm);
    // auto new_shape = v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1});
    // output = make_shared<v1::Reshape>(output, new_shape, true);

    // use naive attention decomposition now
    Output output = attention_softmax(Q, K_compute, V_compute, head_size);

    return {output, K, V};
}


}  // namespace
}  // namespace detail

namespace opset_1 {
ov::OutputVector group_query_attention(const ov::frontend::onnx::Node& node) {
    return detail::group_query_attention_decomposition(
        node.get_ov_inputs(),
        static_cast<int>(node.get_attribute_value<int64_t>("num_heads")),
        static_cast<bool>(node.get_attribute_value<int64_t>("rotary_interleaved", 0)),
        static_cast<int>(node.get_attribute_value<int64_t>("kv_num_heads"))
    );
}

ONNX_OP("GroupQueryAttention", OPSET_SINCE(1), com_microsoft::opset_1::group_query_attention, MICROSOFT_DOMAIN);

}  // namespace opset_1

namespace detail {
namespace {


std::shared_ptr<ov::Node> attention_softmax(Output& Q, 
                                            Output& K,
                                            Output& V,
                                            Output& head_size) {
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    std::shared_ptr<ov::Node> softmax_input = std::make_shared<v0::MatMul>(Q, K, false, true);
    const auto sqrt = std::make_shared<v0::Sqrt>(head_size);
    // (Q x K' + mask) / sqrt(head_size)
    softmax_input = std::make_shared<v1::Divide>(softmax_input, sqrt);
    const auto softmax = std::make_shared<v8::Softmax>(softmax_input, 3);

    // softmax((Q x K' + mask) / sqrt(head_size)) x V
    std::shared_ptr<ov::Node> output = std::make_shared<v0::MatMul>(softmax, V);
    // transpose the result from (batch_size, num_heads, sequence_length, head_size)
    // to (batch_size, sequence_length, num_heads, head_size)
    const auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    output = std::make_shared<v1::Transpose>(output, perm);
    auto new_shape = v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1});
    // reshape the result from (batch_size, sequence_length, num_heads, head_size) to (batch_size, sequence_length,
    // num_heads * head_size)
    output = std::make_shared<v1::Reshape>(output, new_shape, true);

    return output;
}

std::shared_ptr<ov::Node> get_hidden_size(const std::shared_ptr<v3::ShapeOf>& node_shape) {
    // node has shape (batch_size, sequence_length, 3 * hidden_size)
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto hidden_size_x3 = get_elements(node_shape, {2});
    const auto three = v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
    const auto hidden_size = std::make_shared<v1::Divide>(hidden_size_x3, three);
    return hidden_size;
}

ov::NodeVector split_to_QKV(const Output& node,
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
