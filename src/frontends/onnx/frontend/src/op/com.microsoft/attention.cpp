// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/attention.hpp"

#include "default_opset.hpp"
#include "onnx_import/core/null_node.hpp"
#include "ov_models/ov_builders/split.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace detail {
namespace {
NodeVector split_to_QKV(const std::shared_ptr<default_opset::Add>& node,
                        int64_t num_heads,
                        const std::vector<int64_t>& qkv_hidden_sizes);

using NodeTuple = std::tuple<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>>;

NodeTuple get_attention_mask(const OutputVector& op_inputs, bool unidirectional);

std::shared_ptr<ngraph::Node> attention_softmax(const OutputVector& op_inputs,
                                                const std::shared_ptr<ngraph::Node>& Q,
                                                std::shared_ptr<ngraph::Node> K,
                                                std::shared_ptr<ngraph::Node> V,
                                                const std::shared_ptr<ngraph::Node>& attention_mask,
                                                const std::shared_ptr<ngraph::Node>& bin_mask,
                                                const std::shared_ptr<ngraph::Node>& head_size,
                                                bool unidirectional);

std::shared_ptr<ngraph::Node> get_present_state(const std::shared_ptr<ngraph::Node>& K,
                                                const std::shared_ptr<ngraph::Node>& V,
                                                const OutputVector& op_inputs);
}  // namespace
}  // namespace detail

namespace set_1 {
OutputVector attention(const Node& node) {
    auto nodes = node.get_ng_inputs();
    const auto& input = nodes[0];
    const auto& weights = nodes[1];
    const auto& bias = nodes[2];

    // Attention is defined as:
    // Q = input x Wq, K = input x Wk, V = input x Wv
    // attention = softmax((Q x K') / sqrt(head_size)) x V
    //
    // In this operator, Wq, Wk and Wv are combined in a single input 'weights' along the second axis.
    // So the approach here is to do a single big matrix multiply
    // and then split the result into Q, K, V matrices

    auto matmul = std::make_shared<default_opset::MatMul>(input, weights);
    auto add = std::make_shared<default_opset::Add>(matmul, bias);

    const auto num_heads = node.get_attribute_value<int64_t>("num_heads");
    const auto qkv_hidden_sizes = node.get_attribute_value<std::vector<int64_t>>("qkv_hidden_sizes", {});
    const auto split_result = detail::split_to_QKV(add, num_heads, qkv_hidden_sizes);

    bool unidirectional = static_cast<bool>(node.get_attribute_value<int64_t>("unidirectional", 0));
    // mask has values either 0 or -10000 and its shape must be
    // broadcastable to (batch_size, num_heads, sequence_length, past_sequence_length + sequence_length)
    // so it can be added to Q x K' later
    // past_sequence_length can be 0 if 'past' input is not available
    std::shared_ptr<ngraph::Node> attention_mask = nullptr, bin_mask = nullptr;
    std::tie(attention_mask, bin_mask) = detail::get_attention_mask(nodes, unidirectional);

    const auto& Q = split_result[0];
    const auto& K = split_result[1];
    const auto& V = split_result[2];
    const auto& head_size = split_result[3];

    // compute softmax((Q x K' + mask) / sqrt(head_size))
    const auto output = detail::attention_softmax(nodes, Q, K, V, attention_mask, bin_mask, head_size, unidirectional);

    // present = concat(K, V) if 'past' input is unavailable
    // or
    // present = concat(past, K, V)
    const auto present = detail::get_present_state(K, V, nodes);

    return {output, present};
}
}  // namespace set_1

namespace detail {
namespace {

std::shared_ptr<ngraph::Node> get_dimensions(const std::shared_ptr<default_opset::ShapeOf>& shape,
                                             const std::vector<int>& dims) {
    static const auto zero = default_opset::Constant::create(element::i32, Shape{}, {0});
    const auto dims_const = default_opset::Constant::create(element::i32, Shape{dims.size()}, dims);
    return std::make_shared<default_opset::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ngraph::Node> get_dimensions(const std::shared_ptr<ngraph::Node>& node, const std::vector<int>& dims) {
    return get_dimensions(std::make_shared<default_opset::ShapeOf>(node), dims);
}

std::shared_ptr<ngraph::Node> get_hidden_size(const std::shared_ptr<default_opset::ShapeOf>& node_shape) {
    // node has shape (batch_size, sequence_length, 3 * hidden_size)
    const auto zero = default_opset::Constant::create(element::i32, Shape{}, {0});
    const auto hidden_size_x3 = get_dimensions(node_shape, {2});
    const auto three = default_opset::Constant::create(element::i64, Shape{}, {3});
    const auto hidden_size = std::make_shared<default_opset::Divide>(hidden_size_x3, three);
    return hidden_size;
}

NodeVector split_to_QKV(const std::shared_ptr<default_opset::Add>& node,
                        int64_t num_heads,
                        const std::vector<int64_t>& qkv_hidden_sizes) {
    OutputVector split;
    std::shared_ptr<ngraph::Node> head_size = nullptr;
    const auto& node_type = node->get_element_type();
    const auto node_shape = std::make_shared<default_opset::ShapeOf>(node);
    // node has shape (batch_size, sequence_length, 3 * hidden_size)
    // fetch the first two dimensions
    const auto batch_size_seq_len = get_dimensions(node_shape, {0, 1});
    const auto num_heads_node = default_opset::Constant::create(element::i64, Shape{1}, {num_heads});
    if (qkv_hidden_sizes.size() == 0) {
        const auto hidden_size = get_hidden_size(node_shape);
        // head_size = hidden_size / num_heads
        head_size = std::make_shared<default_opset::Divide>(hidden_size, num_heads_node);
        // split the node into 3 even parts Q, K, V with shape (batch_size, sequence_len, hidden_size)
        split = ov::op::util::split(node, 3, 2);
        // and reshape each part to new shape (batch_size, sequence_len, num_heads, head_size)
        auto new_shape =
            std::make_shared<default_opset::Concat>(NodeVector{batch_size_seq_len, num_heads_node, head_size}, 0);
        for (size_t i = 0; i < split.size(); i++) {
            split[i] = std::make_shared<default_opset::Reshape>(split[i], new_shape, false);
        }
        head_size = std::make_shared<default_opset::Convert>(head_size, node_type);
    } else {
        // in this case, weights have shape
        // (input_hidden_size, qkv_hidden_sizes[0] + qkv_hidden_sizes[1] + qkv_hidden_sizes[2])
        // so user specified hidden_sizes for Q, K and V
        NGRAPH_CHECK(qkv_hidden_sizes.size() == 3, "qkv_hidden_sizes attribute needs to have 3 values");
        NGRAPH_CHECK(qkv_hidden_sizes[0] == qkv_hidden_sizes[1],
                     "qkv_hidden_sizes first element should be same as the second");
        // split the node into 3 parts Q, K, V with shapes
        // Q: (batch_size, sequence_len, qkv_hidden_sizes[0])
        // K: (batch_size, sequence_len, qkv_hidden_sizes[1])
        // V: (batch_size, sequence_len, qkv_hidden_sizes[2])
        split = ov::op::util::split(node, qkv_hidden_sizes, 2);
        // and reshape each part to new shape (batch_size, sequence_len, num_heads, head_size)
        for (size_t i = 0; i < split.size(); i++) {
            auto new_shape = std::make_shared<default_opset::Concat>(
                NodeVector{batch_size_seq_len,
                           num_heads_node,
                           default_opset::Constant::create(element::i64, Shape{1}, {qkv_hidden_sizes[i] / num_heads})},
                0);
            split[i] = std::make_shared<default_opset::Reshape>(split[i], new_shape, false);
        }
        float head_size_val = qkv_hidden_sizes[0] > 0 ? static_cast<float>(qkv_hidden_sizes[0]) / num_heads
                                                      : static_cast<float>(qkv_hidden_sizes[2]) / num_heads;
        head_size = default_opset::Constant::create(node_type, Shape{1}, {head_size_val});
    }

    // transpose Q, K and V to (batch_size, num_heads, sequence_len, head_size)
    auto perm = default_opset::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
    auto Q = std::make_shared<default_opset::Transpose>(split[0], perm);
    auto K = std::make_shared<default_opset::Transpose>(split[1], perm);
    auto V = std::make_shared<default_opset::Transpose>(split[2], perm);

    return {Q, K, V, head_size};
}

// This function handles the case when mask_index rank is 1 - so its shape is (batch_size) or (2 * batch_size).
// The returned mask consists of 0 and -10000 and has shape (batch_size, 1, 1, all_seq_len). 'mask_index' input contains
// positions from where the -10000 values start appearing in the final mask per batch (if shape is (batch_size)) or if
// shape is (2 * batch_size), user can define two ranges of -10000 values appearing in the final mask. For example:
//
// batch_size = 3, all_seq_len = 5, mask_index = [2, 4, 3]
// the function returns following mask with shape (3, 1, 1, 5):
// 0,  0,  -10000,  -10000,  -10000
// 0,  0,       0,       0,  -10000
// 0,  0,       0,  -10000,  -10000
//
// e.g., for batch = 2, -10000 values appear within range [mask_index[2]:5] (or [3:5])
//
// Another example, but with mask_index shape (2 * batch_size)
// batch_size = 3, all_seq_len = 5, mask_index = [2, 4, 3, 1, 2, 2]
// the function returns following mask with shape (3, 1, 1, 5):
// -10000,       0,  -10000,  -10000,  -10000
// -10000,  -10000,       0,       0,  -10000
// -10000,  -10000,       0,  -10000,  -10000
//
// e.g., for batch = 1, -10000 values appear within two ranges [0, mask_index[4]] and [mask_index[1]:5] (or [0:2],[4:5])
//
//
// This is how it's done with nGraph operations:
//
//  First the 'base' is generated by range + broadcast:
//     base = range(0, all_seq_len)
//     base = broadcast(base, shape=(batch_size, all_seq_len))
//
//  With batch_size = 3 and all_seq_len = 5, 'base' looks as follows:
//       [[0, 1, 2, 3, 4],
//        [0, 1, 2, 3, 4],
//        [0, 1, 2, 3, 4]]
//
//  Next step is to reshape mask_index:
//     mask_index = reshape(mask_index, shape=(-1, batch_size))
//
//  With the second example above (mask_index = [2, 4, 3, 1, 2, 2]), now it looks like:
//     mask_index = [[2, 4, 3],
//                 [1, 2, 2]]
//
//  Now we get the first row and reshape it to (batch_size, 1) to have indices laid out in column:
//     tail_range_indices = gather(mask_index, indices=[0], axis=0)  # tail_range_indices = [2, 4, 3]
//     tail_range_indices = reshape(tail_range_indices, shape=(batch_size, 1)
//     # tail_range_indices = [[2],
//     #                       [4],
//     #                       [3]]
//
//  Then the base is compared with the indices
//     tail_range_mask = base >= tail_range_indices
//
//  Thanks to autobroadcast in elementwise operators, the comparison conceptually happens between:
//       [[0, 1, 2, 3, 4],      [[2, 2, 2, 2, 2],
//        [0, 1, 2, 3, 4],  >=   [4, 4, 4, 4, 4],
//        [0, 1, 2, 3, 4]]       [3, 3, 3, 3, 3]]
//
//   and the result is:
//               [[0, 0, 1, 1, 1],
//                [0, 0, 0, 0, 1],
//                [0, 0, 0, 1, 1]]
//
// So we get the final tail range mask by multiplying this by -10000
//
// Similarly we process with head range - we fetch the second row from reshaped mask_index,
// compare it with 'base' (but with 'Less' operator instead of 'GreaterEqual') and combine it
// with tail_range_mask.
//
// Handling both mask_index variants (so (batch_size) and (2 * batch_size)) is tricky since we don't
// know its dimensions upfront. So we compute both variants and use Select operator to select
// the right one in the runtime (unless it gets constantfolded before).
std::shared_ptr<ngraph::Node> attention_mask_from_indices(const Output<ngraph::Node>& mask_index,
                                                          const element::Type_t& type,
                                                          const std::shared_ptr<ngraph::Node>& batch_size,
                                                          const std::shared_ptr<ngraph::Node>& all_seq_len) {
    const auto zero = default_opset::Constant::create(element::i64, Shape{}, {0});
    const auto one = default_opset::Constant::create(element::i64, Shape{}, {1});
    const auto stop = std::make_shared<default_opset::Squeeze>(all_seq_len, zero);
    std::shared_ptr<ngraph::Node> base =
        std::make_shared<default_opset::Range>(zero, stop, one, mask_index.get_element_type());
    const auto target_shape = std::make_shared<default_opset::Concat>(NodeVector{batch_size, all_seq_len}, 0);
    // broadcast 'base' to (batch_size, all_seq_len)
    base = std::make_shared<default_opset::Broadcast>(base, target_shape);
    const auto indices_shape = std::make_shared<default_opset::Concat>(
        NodeVector{default_opset::Constant::create(element::i64, Shape{1}, {-1}), batch_size},
        0);
    std::shared_ptr<ngraph::Node> indices = std::make_shared<default_opset::Reshape>(mask_index, indices_shape, false);
    // fetch first row from indices
    std::shared_ptr<ngraph::Node> tail_range_indices = std::make_shared<default_opset::Gather>(indices, zero, zero);
    tail_range_indices =
        std::make_shared<default_opset::Reshape>(tail_range_indices,
                                                 default_opset::Constant::create(element::i32, Shape{2}, {-1, 1}),
                                                 false);
    const auto greater_eq = std::make_shared<default_opset::GreaterEqual>(base, tail_range_indices);
    std::shared_ptr<ngraph::Node> tail_range_mask =
        std::make_shared<default_opset::Multiply>(std::make_shared<default_opset::Convert>(greater_eq, type),
                                                  default_opset::Constant::create(type, Shape{}, {-10000}));
    tail_range_mask =
        std::make_shared<default_opset::Unsqueeze>(tail_range_mask,
                                                   default_opset::Constant::create(element::i64, Shape{2}, {1, 2}));

    const auto gather_index =
        std::make_shared<default_opset::FloorMod>(default_opset::Constant::create(element::i64, Shape{}, {1}),
                                                  get_dimensions(indices, {0}));
    // fetch indices from the second row (or first if not available)
    std::shared_ptr<ngraph::Node> head_range_indices =
        std::make_shared<default_opset::Gather>(indices, gather_index, zero);
    head_range_indices =
        std::make_shared<default_opset::Reshape>(head_range_indices,
                                                 default_opset::Constant::create(element::i32, Shape{2}, {-1, 1}),
                                                 false);
    const auto less = std::make_shared<default_opset::Less>(base, head_range_indices);
    std::shared_ptr<ngraph::Node> mask = std::make_shared<default_opset::LogicalOr>(less, greater_eq);
    mask = std::make_shared<default_opset::Multiply>(std::make_shared<default_opset::Convert>(mask, type),
                                                     default_opset::Constant::create(type, Shape{}, {-10000}));
    // reshape from (batch_size, all_seq_len) to (batch_size, 1, 1, all_seq_len)
    mask = std::make_shared<default_opset::Unsqueeze>(mask,
                                                      default_opset::Constant::create(element::i64, Shape{2}, {1, 2}));

    const auto mask_index_first_dim = get_dimensions(mask_index.get_node_shared_ptr(), {0});
    // compare mask_index.shape[0] with batch_size value
    // if they're equal - select tail_range_mask
    // else select full mask
    mask = std::make_shared<default_opset::Select>(
        std::make_shared<default_opset::Equal>(batch_size, mask_index_first_dim),
        tail_range_mask,
        mask);

    return mask;
}

// Prepare unidirectional_mask like it's done in
// https://github.com/microsoft/onnxruntime/blob/851554536ca8185b3413ee57449ea5ac93370193/onnxruntime/contrib_ops/cpu/bert/attention_helper.h#L87-L96
//
// Function returns two masks - one attention mask with values 0 or -10000 with shape (seq_len, all_seq_len),
// the second one is a binary mask where it has 0 on positions where attention mask has -10000 values and 1 otherwise.
//
// For example:
// seq_len = 4, all_seq_len = 7, past_seq_len = 3. Returned attention mask has shape (4, 7) and contains:
// 0  0  0  0  -10000  -10000  -10000
// 0  0  0  0       0  -10000  -10000
// 0  0  0  0       0       0  -10000
// 0  0  0  0       0       0       0
//
// Returned binary mask has the shape (4, 7) and following values:
// 1  1  1  1  0  0  0
// 1  1  1  1  1  0  0
// 1  1  1  1  1  1  0
// 1  1  1  1  1  1  1
//
// Binary mask is used later before softmax to achieve
// https://github.com/microsoft/onnxruntime/blob/851554536ca8185b3413ee57449ea5ac93370193/onnxruntime/contrib_ops/cpu/bert/attention_cpu_base.h#L158-L166
//
// The approach used to generate those masks is similar to one from attention_mask_from_indices function (see comments
// there).
NodeTuple unidirectional_mask(const element::Type_t& type,
                              const std::shared_ptr<ngraph::Node>& seq_len,
                              const std::shared_ptr<ngraph::Node>& all_seq_len,
                              const std::shared_ptr<ngraph::Node>& past_seq_len) {
    const auto zero = default_opset::Constant::create(element::i64, Shape{}, {0});
    const auto one = default_opset::Constant::create(element::i64, Shape{}, {1});
    const auto stop = std::make_shared<default_opset::Squeeze>(all_seq_len, zero);
    std::shared_ptr<ngraph::Node> bin_mask = std::make_shared<default_opset::Range>(zero, stop, one, element::i32);
    auto target_shape = std::make_shared<default_opset::Concat>(NodeVector{seq_len, all_seq_len}, 0);
    bin_mask = std::make_shared<default_opset::Broadcast>(bin_mask, target_shape);
    auto start =
        std::make_shared<default_opset::Squeeze>(std::make_shared<default_opset::Add>(past_seq_len, one), zero);
    auto end = std::make_shared<default_opset::Squeeze>(std::make_shared<default_opset::Add>(all_seq_len, one), zero);
    auto indices = std::make_shared<default_opset::Unsqueeze>(
        std::make_shared<default_opset::Range>(start, end, one, element::i32),
        default_opset::Constant::create(element::i32, Shape{1}, {1}));
    bin_mask = std::make_shared<default_opset::GreaterEqual>(bin_mask, indices);
    std::shared_ptr<ngraph::Node> attention_mask =
        std::make_shared<default_opset::Multiply>(std::make_shared<default_opset::Convert>(bin_mask, type),
                                                  default_opset::Constant::create(type, Shape{}, {-10000}));
    bin_mask = std::make_shared<default_opset::Convert>(std::make_shared<default_opset::LogicalNot>(bin_mask), type);
    return NodeTuple{attention_mask, bin_mask};
}

// This is the easiest variant of 'mask_index' input - the input consists of 0 or 1 values
// and we transform them to:
// * -10000 for positions where mask_index == 0
// * 0 for positions where mask_index == 1
//
// It handles mask_index with shapes:
// (batch_size, past_sequence_length + sequence_length) or
// (batch_size, sequence_length, past_sequence_length + sequence_length)
//
// Shape (batch_size, 1, max_sequence_length, max_sequence_length) is not supported in onnxruntime:
// https://github.com/microsoft/onnxruntime/blob/851554536ca8185b3413ee57449ea5ac93370193/onnxruntime/contrib_ops/cpu/bert/attention_helper.h#L78
std::shared_ptr<ngraph::Node> raw_mask(const Output<ngraph::Node>& mask_index,
                                       Dimension::value_type mask_rank,
                                       const element::Type_t& type) {
    std::shared_ptr<ngraph::Node> mask = std::make_shared<default_opset::Convert>(mask_index, type);
    mask = std::make_shared<default_opset::Convert>(mask, type);
    mask = std::make_shared<default_opset::Subtract>(default_opset::Constant::create(type, Shape{}, {1}), mask);
    mask = std::make_shared<default_opset::Multiply>(mask, default_opset::Constant::create(type, Shape{}, {-10000}));
    switch (mask_rank) {
    // Handle mask_index with (batch_size, past_sequence_length + sequence_length) shape
    // Reshape it to (batch_size, 1, 1, past_sequence_length + sequence_length)
    case 2:
        mask = std::make_shared<default_opset::Reshape>(
            mask,
            default_opset::Constant::create(element::i64, Shape{4}, {0, 1, 1, -1}),
            true);
        break;
    // Handle mask_index with (batch_size, sequence_length, past_sequence_length + sequence_length) shape
    // Reshape it to (batch_size, 1, sequence_length, past_sequence_length + sequence_length)
    case 3:
        mask = std::make_shared<default_opset::Reshape>(
            mask,
            default_opset::Constant::create(element::i64, Shape{4}, {0, 1, 0, -1}),
            true);
        break;
    }
    return mask;
}

bool is_past_input_available(const OutputVector& op_inputs) {
    return op_inputs.size() > 4 && !ngraph::op::is_null(op_inputs[4]);
}

NodeTuple get_attention_mask(const OutputVector& op_inputs, bool unidirectional) {
    const auto zero = default_opset::Constant::create(element::i64, Shape{1}, {0});
    const auto one = default_opset::Constant::create(element::i64, Shape{1}, {1});

    std::shared_ptr<ngraph::Node> past_seq_len;
    // get the value of past_sequence_length
    if (is_past_input_available(op_inputs)) {
        const auto& past = op_inputs[4];
        // 'past' node has shape (2, batch_size, num_heads, past_sequence_length, head_size)
        past_seq_len = get_dimensions(past.get_node_shared_ptr(), {3});
    } else {
        past_seq_len = zero;
    }

    // 'input' node has shape (batch_size, sequence_length, input_hidden_size)
    auto input_shape = std::make_shared<default_opset::ShapeOf>(op_inputs[0]);
    auto seq_len = get_dimensions(input_shape, {1});
    auto all_seq_len = std::make_shared<default_opset::Add>(seq_len, past_seq_len);
    const auto& type = op_inputs[0].get_element_type();
    std::shared_ptr<ngraph::Node> attention_mask = nullptr;
    std::shared_ptr<ngraph::Node> bin_mask = nullptr;
    if (unidirectional) {
        std::tie(attention_mask, bin_mask) = unidirectional_mask(type, seq_len, all_seq_len, past_seq_len);
    }
    if (op_inputs.size() > 3 && !ngraph::op::is_null(op_inputs[3])) {
        const auto& mask_index = op_inputs[3];
        NGRAPH_CHECK(mask_index.get_element_type() == element::i32, "'mask_index' type must be int32");
        auto batch_size = get_dimensions(input_shape, {0});
        const auto mask_rank = mask_index.get_partial_shape().rank();
        NGRAPH_CHECK(mask_rank.is_static(), "'mask_index' rank must be static");
        auto mask_rank_val = mask_rank.get_length();
        std::shared_ptr<ngraph::Node> mask;
        if (mask_rank_val == 1) {
            // case when mask_index has shape (batch_size) or (2 * batch_size)
            // so it contains positions that specify how mask should be generated
            mask = attention_mask_from_indices(mask_index, type, batch_size, all_seq_len);
        } else if (mask_rank_val < 4) {
            mask = raw_mask(mask_index, mask_rank.get_length(), type);
        } else {
            NGRAPH_CHECK(false, "mask_index with rank " + std::to_string(mask_rank_val) + " is not supported");
        }
        // add the mask with unidirectional mask if available
        if (attention_mask) {
            attention_mask = std::make_shared<default_opset::Add>(attention_mask, mask);
        } else {
            attention_mask = mask;
        }
    }
    return NodeTuple{attention_mask, bin_mask};
}

// Compute softmax(Q x K' / sqrt(head_size)) x V
std::shared_ptr<ngraph::Node> attention_softmax(const OutputVector& op_inputs,
                                                const std::shared_ptr<ngraph::Node>& Q,
                                                std::shared_ptr<ngraph::Node> K,
                                                std::shared_ptr<ngraph::Node> V,
                                                const std::shared_ptr<ngraph::Node>& attention_mask,
                                                const std::shared_ptr<ngraph::Node>& bin_mask,
                                                const std::shared_ptr<ngraph::Node>& head_size,
                                                bool unidirectional) {
    auto zero = default_opset::Constant::create(element::i64, Shape{}, {0});
    if (is_past_input_available(op_inputs)) {
        // concat past K and V with present ones
        const auto& past = op_inputs[4];
        // 'past' input has two matrices K and V with shape (1, batch_size, num_heads, past_sequence_length, head_size)
        // concatenated along first axis to a single
        // (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)
        // so we need to split it into two parts, remove first dimension from each part and concatenate first part
        // with current K and second part with current V
        const auto split = ov::op::util::split(past, 2, 0);
        const auto past_K = std::make_shared<default_opset::Squeeze>(split[0], zero);
        K = std::make_shared<default_opset::Concat>(NodeVector{past_K, K}, 2);
        const auto past_V = std::make_shared<default_opset::Squeeze>(split[1], zero);
        V = std::make_shared<default_opset::Concat>(NodeVector{past_V, V}, 2);
    }
    // perform Q x K'
    std::shared_ptr<ngraph::Node> softmax_input = std::make_shared<default_opset::MatMul>(Q, K, false, true);
    // Q x K' + mask
    if (attention_mask) {
        if (unidirectional) {
            // Perform the equivalent of
            // https://github.com/microsoft/onnxruntime/blob/851554536ca8185b3413ee57449ea5ac93370193/onnxruntime/contrib_ops/cpu/bert/attention_cpu_base.h#L158-L166
            // For positions where unidirectional_mask has -10000 values - attention_mask is moved to softmax input
            softmax_input = std::make_shared<default_opset::Multiply>(softmax_input, bin_mask);
        }
        softmax_input = std::make_shared<default_opset::Add>(softmax_input, attention_mask);
    }
    const auto sqrt = std::make_shared<default_opset::Sqrt>(head_size);
    // (Q x K' + mask) / sqrt(head_size)
    softmax_input = std::make_shared<default_opset::Divide>(softmax_input, sqrt);
    // handle 'extra_add' input
    if (op_inputs.size() > 5 && !ngraph::op::is_null(op_inputs[5])) {
        NGRAPH_CHECK(!is_past_input_available(op_inputs),
                     "Cannot use both 'past' and 'extra_add' inputs in the same node");
        const auto& extra_add = op_inputs[5];
        softmax_input = std::make_shared<default_opset::Add>(softmax_input, extra_add);
    }
    // softmax((Q x K' + mask) / sqrt(head_size))
    const auto softmax = std::make_shared<default_opset::Softmax>(softmax_input, 3);

    // softmax((Q x K' + mask) / sqrt(head_size)) x V
    std::shared_ptr<ngraph::Node> output = std::make_shared<default_opset::MatMul>(softmax, V);
    // transpose the result from (batch_size, num_heads, sequence_length, head_size)
    // to (batch_size, sequence_length, num_heads, head_size)
    const auto perm = default_opset::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
    output = std::make_shared<default_opset::Transpose>(output, perm);
    auto new_shape = default_opset::Constant::create(element::i32, Shape{3}, {0, 0, -1});
    // reshape the result from (batch_size, sequence_length, num_heads, head_size) to (batch_size, sequence_length,
    // num_heads * head_size)
    output = std::make_shared<default_opset::Reshape>(output, new_shape, true);

    return output;
}

// Make present state from K and V matrices by reshaping them from:
// (batch_size, num_heads, sequence_length, head_size) to (1, batch_size, num_heads, sequence_length, head_size)
// and concatenating them along first axis to make 'present' output.
// If fifth input ('past') is available, it gets concatenated with 'present' output along fourth axis.
std::shared_ptr<ngraph::Node> get_present_state(const std::shared_ptr<ngraph::Node>& K,
                                                const std::shared_ptr<ngraph::Node>& V,
                                                const OutputVector& op_inputs) {
    auto zero = default_opset::Constant::create(element::i64, Shape{1}, {0});
    // expand K shape (batch_size, num_heads, sequence_length, head_size) to
    // (1, batch_size, num_heads, sequence_length, head_size)
    auto K_unsqueezed = std::make_shared<default_opset::Unsqueeze>(K, zero);
    // similarly expand V shape
    auto V_unsqueezed = std::make_shared<default_opset::Unsqueeze>(V, zero);

    // add padding in case K and V have different shapes (it happens when used provided uneven qkv_hidden_sizes)
    // if the shapes are equal (so padding will be zero), Pad gets eliminated in NopElimination pass
    const auto K_shape = std::make_shared<default_opset::ShapeOf>(K_unsqueezed);
    const auto V_shape = std::make_shared<default_opset::ShapeOf>(V_unsqueezed);
    const auto K_pads_end =
        std::make_shared<default_opset::Maximum>(std::make_shared<default_opset::Subtract>(V_shape, K_shape), zero);
    const auto V_pads_end =
        std::make_shared<default_opset::Maximum>(std::make_shared<default_opset::Subtract>(K_shape, V_shape), zero);
    const auto pads_begin =
        std::make_shared<default_opset::Broadcast>(zero, std::make_shared<default_opset::ShapeOf>(K_shape));
    const auto K_padded =
        std::make_shared<default_opset::Pad>(K_unsqueezed, pads_begin, K_pads_end, ngraph::op::PadMode::CONSTANT);
    const auto V_padded =
        std::make_shared<default_opset::Pad>(V_unsqueezed, pads_begin, V_pads_end, ngraph::op::PadMode::CONSTANT);

    // concat key and value tensors along first axis to make 'present' state
    // after that operation, 'present' has shape (2, batch_size, num_heads, sequence_length, head_size)
    std::shared_ptr<ngraph::Node> present = std::make_shared<default_opset::Concat>(NodeVector{K_padded, V_padded}, 0);
    if (is_past_input_available(op_inputs)) {
        const auto& past = op_inputs[4];
        // concat 'past' to 'present' output along fourth axis
        // after that operation, 'present' has shape:
        // (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)
        present = std::make_shared<default_opset::Concat>(OutputVector{past, present}, 3);
    }
    return present;
}
}  // namespace
}  // namespace detail
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
