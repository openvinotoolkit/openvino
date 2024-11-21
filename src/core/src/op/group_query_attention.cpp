// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <optional>
#include <functional>

#include "openvino/op/group_query_attention.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/constant.hpp"


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
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_update.hpp"
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
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace op {

namespace detail {
namespace {

int64_t get_head_size (const PartialShape& input_shape, int num_heads, int kv_num_heads) {
    return input_shape[2].get_length() / (num_heads + kv_num_heads*2);
}


std::vector<int64_t> get_qkv_sizes (const PartialShape& input_shape, int num_heads, int kv_num_heads) {
    int64_t per_head_size = get_head_size(input_shape, num_heads, kv_num_heads);
    const std::vector<int64_t> qkv_sizes = {num_heads*per_head_size, kv_num_heads*per_head_size, kv_num_heads*per_head_size};
    return qkv_sizes;
}

}
}

Output<Node> GroupQueryAttentionExtension::null() {
    return v0::Constant::create(element::f32, Shape{0}, {});    // particular type and shape do not matter
}

bool GroupQueryAttentionExtension::is_null(const Output<Node> output) {
    if(std::dynamic_pointer_cast<v0::Constant>(output.get_node_shared_ptr())) {
        return output.get_shape().size() == 1 && output.get_element_type() == element::f32; // should match exactly with what we are creating in null function
    }
}



GroupQueryAttentionExtension::GroupQueryAttentionExtension(const ov::OutputVector& args, unsigned int num_heads, unsigned int kv_num_heads, bool rotary_interleaved) :
    ov::op::Op(args),
    m_num_heads(num_heads),
    m_kv_num_heads(kv_num_heads),
    m_rotary_interleaved(rotary_interleaved) {
    constructor_validate_and_infer_types();
}

void GroupQueryAttentionExtension::validate_and_infer_types() {
    OV_OP_SCOPE(GroupQueryAttentionExtension_validate_and_infer_types);
    PartialShape input_shape = get_input_partial_shape(0);
    Dimension batch_size = input_shape[0];
    Dimension sequence_len = input_shape[1];
    Dimension head_size;
    if(is_null(input_value(1)) && is_null(input_value(2))) {
        head_size = detail::get_head_size(input_shape, m_num_heads, m_kv_num_heads);
    } else {
        head_size = input_shape[2].get_length() / m_num_heads;
    }
    Dimension output_kv_len;
    PartialShape kv_past_shape = get_input_partial_shape(3);
    // FIXME: Original GQA spec depends on the identical tensor set for input/output, but we cannot know it in advance, hence we base on sequence dimension static/dynamic
    if(kv_past_shape[2].is_dynamic()) {
        output_kv_len = kv_past_shape[2] + sequence_len;
    } else {
        output_kv_len = kv_past_shape[2];
    }
    auto element_type = get_input_element_type(0);
    set_output_type(0, element_type, PartialShape{batch_size, sequence_len, head_size*m_num_heads});
    set_output_type(1, element_type, PartialShape{batch_size, m_kv_num_heads, output_kv_len, head_size});
    set_output_type(2, element_type, PartialShape{batch_size, m_kv_num_heads, output_kv_len, head_size});
}

std::shared_ptr<ov::Node> GroupQueryAttentionExtension::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<GroupQueryAttentionExtension>(new_args, m_num_heads, m_kv_num_heads, m_rotary_interleaved);
}

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
using v3::ScatterElementsUpdate;
using v3::ScatterUpdate;
using v15::Squeeze;
using v1::Less;
using v1::LessEqual;
using v8::Gather;
using v1::VariadicSplit;
using Output = ov::Output<ov::Node>;
using std::make_shared;

Output get_elements(const Output& shape, const std::vector<int>& dims) {
    static const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

// FIXME: Reuse the same function from file attention.cpp, but it requires a bit of adaptation -- I have redesigned part of the inputs a bit here and in the helper functions below
std::shared_ptr<ov::Node> get_hidden_size(const std::shared_ptr<v3::ShapeOf>& node_shape) {
    // node has shape (batch_size, sequence_length, 3 * hidden_size)
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto hidden_size_x3 = get_elements(node_shape, {2});
    const auto three = v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
    const auto hidden_size = std::make_shared<v1::Divide>(hidden_size_x3, three);
    return hidden_size;
}

// make split functions is a copy-past from ONNX FE. TODO: move it to one place
OutputVector make_split(const Output& value, const std::vector<int64_t>& split_lengths, int64_t axis) {
    const auto axis_node = Constant::create(ov::element::i64, Shape{}, {axis});
    const auto split_lengths_node =
        Constant::create(ov::element::i64, Shape{split_lengths.size()}, split_lengths);
    const auto variadic_split = make_shared<VariadicSplit>(value, axis_node, split_lengths_node);

    return variadic_split->outputs();
}

OutputVector make_split(const Output& value, int64_t num_splits, int64_t axis) {
    const auto axis_node = Constant::create(ov::element::i64, Shape{}, {axis});
    const auto split = make_shared<Split>(value, axis_node, num_splits);

    return split->outputs();
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
        split = make_split(node, 3, 2);
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
        OPENVINO_ASSERT(qkv_hidden_sizes.size() == 3, "qkv_hidden_sizes attribute needs to have 3 values");
        OPENVINO_ASSERT(qkv_hidden_sizes[0] == qkv_hidden_sizes[1],
                                "qkv_hidden_sizes first element should be same as the second");
        // split the node into 3 parts Q, K, V with shapes
        // Q: (batch_size, sequence_len, qkv_hidden_sizes[0])
        // K: (batch_size, sequence_len, qkv_hidden_sizes[1])
        // V: (batch_size, sequence_len, qkv_hidden_sizes[2])
        split = make_split(node, qkv_hidden_sizes, 2);
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


Output get_dimensions(const Output& node, const std::vector<int>& dims) {
    return get_elements(std::make_shared<v3::ShapeOf>(node, element::i32), dims);
}

Output squeeze_1d(const Output& x) {
    return make_shared<Squeeze>(x, Constant::create(element::i32, Shape{0}, {1}));
}

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

    Output zero = Constant::create(element::i32, Shape{}, {0});
    Output one = Constant::create(element::i32, Shape{}, {1});

    // cut for the current sequence length
    // TODO: Avoid using Slices because they lead to dynamic dimensions
    // Output cos = make_shared<Slice>(cos_cache, pos_id_begin, pos_id_end, step, zero);
    // Output sin = make_shared<Slice>(sin_cache, pos_id_begin, pos_id_end, step, zero);

    Output seq_len = get_dimensions(x, {2});
    Output position_ids = make_shared<Range>(zero, squeeze_1d(seq_len), one, element::i32);
    position_ids = make_shared<Add>(pos_id_begin, position_ids);
    Output cos = make_shared<Gather>(cos_cache, position_ids, zero);
    Output sin = make_shared<Gather>(sin_cache, position_ids, zero);

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

// The next methods are variants of cache management. Each return a pair of outputs: {kv_to_be_passed_to_sdpa, kv_to_be_returned_from_decomp}

OutputVector concat_cache_generic(const Output& past, const Output& current) {
    const Output concated = make_shared<Concat>(ov::OutputVector{past, current}, 2);     // 2 is the dimension index that corresponds to sequence len
    return {concated, concated};
}

OutputVector update_cache_generic(const Output& past, const Output& current, const Output& past_len) {
    Output update_len = get_dimensions(current, {2});
    Output update_end = make_shared<Add>(past_len, update_len);
    Output update_indices = make_shared<Range>(squeeze_1d(past_len), squeeze_1d(update_end), Constant::create(element::i32, Shape{}, {1}), element::i32);
    Output updated = make_shared<ScatterUpdate>(past, update_indices, current, Constant::create(element::i32, Shape{1}, {2}));    // 2 is the dimension index that corresponds to sequence len
    return {updated, updated};
}

OutputVector update_cache_npuw(const Output& past, const Output& current, const Output& past_len) {
    return {concat_cache_generic(past, current)[0], current};
}

OutputVector update_cache_seu(const Output& past, const Output& current, const Output& past_len) {
    Output update_len = get_dimensions(current, {2});
    const auto one = Constant::create(element::i32, Shape{}, {1});
    Output update_end = make_shared<Add>(past_len, update_len);
    Output update_indices = make_shared<Range>(squeeze_1d(past_len), squeeze_1d(update_end), one, element::i32);
    auto shape_of = make_shared<ShapeOf>(current, element::i32);
    auto unsqueezed_indices = make_shared<Unsqueeze>(update_indices, Constant::create(element::i32, Shape{3}, {0,1,3}));
    auto broadcast_indices = make_shared<Broadcast>(unsqueezed_indices, shape_of);
    Output seu = make_shared<ScatterElementsUpdate>(past, broadcast_indices, current, Constant::create(element::i32, Shape{1}, {2}));
    return {seu, seu};
}

Output attn_mask_npuw(
    const Output& past_seq_len,
    const Output& current_seq_len,
    const Output& total_seq_len,
    const Output& kv_tensors_seq_len
) {
    // make casual 2D attention mask based on pattern [1, 1, ..., 1, 0, 0, ..., 0, 1, 1, ..., 1], where
    //    - the first block of 1s has lenght equal to `past_seq_len`, usually dynamic
    //    - the last block of 1s has length equal to `current_seq_len` (usually = 1 in case of generate iteration), usually static
    //    - the total length of the pattern is `kv_tensor_seq_len`, usually static

    const Output zero = Constant::create(ov::element::i32, ov::Shape{}, {0});
    const Output one = Constant::create(ov::element::i32, ov::Shape{}, {1});

    const Output past_range = make_shared<Range>(zero, squeeze_1d(total_seq_len), one, element::i32);
    Output past_mask = make_shared<Less>(past_range, past_seq_len);  // [1, 1, ...., 1, 0, 0, ..., 0]
    past_mask = make_shared<Broadcast>(past_mask, make_shared<Concat>(OutputVector{current_seq_len, total_seq_len}, 0));  // replicated current_seq_len times

    const Output curr_range = make_shared<Range>(zero, squeeze_1d(current_seq_len), one, element::i32);
    const Output curr_mask = make_shared<LessEqual>(curr_range, make_shared<Unsqueeze>(curr_range, Constant::create(element::i32, Shape{}, {1})));

    return make_shared<Concat>(ov::OutputVector{past_mask, curr_mask}, 1);
}

using attn_mask_callback = std::function<Output(
        const Output& past_seq_len,     // occupied part of KV cache, 1D tensor with one or multiple elements depending on the current level of support of unaligned sequences in the batch
        const Output& current_seq_len,  // number of currently processed tokens, usually 1 in the normal generate iteration (not prefill)
        const Output& total_seq_len,
        const Output& kv_tensors_seq_len   // the total sequence length of KV tensors that come to SDPA to generate a mask of the correct shape even if KV is partially filled
    )>;

ov::OutputVector group_query_attention_decomposition(
    const ov::OutputVector& inputs,
    int num_heads,
    bool rotary_interleaved,
    int kv_num_heads,
    std::function<OutputVector(const Output& past, const Output& current)> concat_cache,
    std::function<OutputVector(const Output& past, const Output& current, const Output& past_len)> update_cache,
    attn_mask_callback attn_mask = attn_mask_callback()  // attn_mask generator for SDPA, if not provided, the attn_mask input is not used and causal flag is set
) {
    const auto& input = inputs[0];

    Output Q, K, V, head_size;

    if(GroupQueryAttentionExtension::is_null(inputs[1]) || GroupQueryAttentionExtension::is_null(inputs[2])) {
        auto qkv_sizes = get_qkv_sizes(input.get_partial_shape(), num_heads, kv_num_heads);
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

    const auto& past_K = inputs[3];
    const auto& past_V = inputs[4];
    const auto& seqlens_k = inputs[5];
    //const auto& total_sequence_length = inputs[6];  // use it for checking dimensions only, it should be deduced from other parameters and shapes
    const auto& cos = inputs[7];
    const auto& sin = inputs[8];

    // FIXME: It works only when KV cache is dynamically growing and doesn't have unused space inside. So it is not compatible with statically-shaped KV cache.
    // const auto past_seq_len = detail::get_dimensions(past_K, {0});
    // TODO: GQA spec is not compatible with test model. Spec supposes 1D tensor, in the test model we have 2D tensor, flattening to work in both cases.

    // FIXME: Unaligned elements in KV cache are not supported.
    // We just get the first of the seq lens as a common value for all past sequences ignoring others, under assumption that they are all the same
    const auto& past_seq_len = get_elements(std::make_shared<v1::Reshape>(seqlens_k, v0::Constant::create(element::i32, Shape{1}, {-1}), false), {0});
    const auto& curr_seq_len = get_dimensions(Q, {2});
    const auto& past_tensor_size = get_dimensions(past_V, {2});
    const Output& total_sequence_len = std::make_shared<Add>(past_seq_len, curr_seq_len);

    Q = rope(Q, cos, sin, rotary_interleaved, head_size, past_seq_len, total_sequence_len);
    K = rope(K, cos, sin, rotary_interleaved, head_size, past_seq_len, total_sequence_len);

    OutputVector K_tuple, V_tuple;

    if(past_K.get_partial_shape()[2].is_dynamic()) {
        K_tuple = concat_cache(past_K, K);
    } else {
        K_tuple = update_cache(past_K, K, past_seq_len);
    }

    if(past_V.get_partial_shape()[2].is_dynamic()) {
        V_tuple = concat_cache(past_V, V);
    } else {
        V_tuple = update_cache(past_V, V, past_seq_len);
    }

    K = K_tuple[0];
    V = V_tuple[0];

    K = broadcast_groups(K, kv_num_heads, num_heads);
    V = broadcast_groups(V, kv_num_heads, num_heads);

    // FIXME: Unaligned batch of sequences is not supported. All past key-value are assumed to have the same length.
    // That means all input sequence lengths should be the same and match input.shape[2]
    // We do not check that here because it depends on runtime values.
    // If we want to implement not aligned batch of dimensions we have to form not uniform causal mask for attention that
    // adds a significant porition of the code.

    // FIXME: The same tensor at input/output of past/preset K and V are not supported.
    // It requires more complex tensor manipulations that are introduce overhead into pure tensor-value data flow and should be implemented if we really have demand for that.
    // Also inplace KV-cache modification logic is not supported efficiently in any plugins (CPU, GPU and NPU).

    OutputVector sdpa_inputs{Q, K, V};

    if(attn_mask) {
        sdpa_inputs.push_back(attn_mask(past_seq_len, curr_seq_len, past_tensor_size, get_dimensions(V, {2})));
    }

    auto output = make_shared<v13::ScaledDotProductAttention>(sdpa_inputs, !attn_mask);

    return {output, K_tuple[1], V_tuple[1]};
}


}  // namespace
}  // namespace detail


ov::OutputVector group_query_attention_decomposition(std::shared_ptr<GroupQueryAttentionExtension> gqa) {
    return detail::group_query_attention_decomposition(
        gqa->input_values(),
        gqa->get_num_heads(),
        gqa->get_rotary_interleaved(),
        gqa->get_kv_num_heads(),
        detail::concat_cache_generic,
        detail::update_cache_npuw,
        detail::attn_mask_npuw
    );
}


}  // namespace op
}  // namespace ov
