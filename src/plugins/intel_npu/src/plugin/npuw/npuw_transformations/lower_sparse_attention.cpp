// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lower_sparse_attention.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "detect_causal_mask.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/sparse_attention.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v6 = ov::op::v6;
namespace v8 = ov::op::v8;
namespace v13 = ov::op::v13;
namespace v17 = ov::op::v17;

struct Builder {
    ov::NodeVector nodes;

    template <typename T, typename... Args>
    std::shared_ptr<T> make(Args&&... args) {
        auto node = std::make_shared<T>(std::forward<Args>(args)...);
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<v0::Constant> scalar(int32_t value) {
        auto node = v0::Constant::create(ov::element::i32, ov::Shape{}, {value});
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<v0::Constant> axes(std::vector<int64_t> values) {
        auto node = v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<v0::Constant> indices(const ov::Shape& shape, const std::vector<int32_t>& values) {
        auto node = v0::Constant::create(ov::element::i32, shape, values);
        nodes.push_back(node);
        return node;
    }

    std::shared_ptr<v0::Constant> real(float value) {
        auto node = v0::Constant::create(ov::element::f32, ov::Shape{}, {value});
        nodes.push_back(node);
        return node;
    }
};

bool lower(const std::shared_ptr<v17::SparseAttention>& attention,
           const std::shared_ptr<ov::Node>& physical_attention_mask) {
    const auto& query_shape = attention->get_input_partial_shape(0);
    const auto& key_shape = attention->get_input_partial_shape(1);
    const auto& value_shape = attention->get_input_partial_shape(2);
    const auto& indices_shape = attention->get_input_partial_shape(3);
    const auto& count_shape = attention->get_input_partial_shape(4);
    if (!attention->get_causal() || attention->get_query_block_size() != 1 || attention->get_input_size() == 8 ||
        !query_shape.is_static() || !key_shape.is_static() || !value_shape.is_static() || !indices_shape.is_static() ||
        !count_shape.is_static() || query_shape.size() != 4 || key_shape.size() != 4 || value_shape.size() != 4 ||
        indices_shape.size() != 4 || count_shape.size() != 3 || query_shape[0].get_length() != 1 ||
        key_shape[0].get_length() != 1 || indices_shape[0].get_length() != 1 || value_shape[0].get_length() != 1 ||
        indices_shape[1].get_length() != 1 || count_shape[0].get_length() != 1 || count_shape[1].get_length() != 1) {
        return false;
    }

    const auto query_length = query_shape[2].get_length();
    const auto key_length = key_shape[2].get_length();
    const auto query_heads = query_shape[1].get_length();
    const auto key_heads = key_shape[1].get_length();
    const auto selection_width = indices_shape[3].get_length();
    const auto block_size = attention->get_key_block_size();
    if (query_length <= 0 || key_length <= 0 || query_length > key_length || query_heads <= 0 || key_heads <= 0 ||
        query_heads % key_heads != 0 || value_shape[1] != key_shape[1] || value_shape[2] != key_shape[2] ||
        selection_width <= 0 || block_size <= 0 || indices_shape[2].get_length() != query_length ||
        count_shape[2].get_length() != query_length || key_length > std::numeric_limits<int32_t>::max() ||
        query_length > std::numeric_limits<int32_t>::max() || selection_width > std::numeric_limits<int32_t>::max() ||
        block_size > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    const auto block_count = (key_length - 1) / block_size + 1;
    if (block_count >= std::numeric_limits<int32_t>::max()) {
        return false;
    }

    Builder b;
    const auto zero = b.scalar(0);
    const auto one = b.scalar(1);
    const auto block = b.scalar(static_cast<int32_t>(block_size));
    const auto sentinel = b.scalar(static_cast<int32_t>(block_count));
    const auto total_keys = b.scalar(static_cast<int32_t>(key_length));
    const auto total_queries = b.scalar(static_cast<int32_t>(query_length));
    const auto batch_and_head = b.axes({0, 1});
    const auto indices =
        b.make<v0::Convert>(b.make<v0::Squeeze>(attention->input_value(3), batch_and_head), ov::element::i32);
    const auto counts =
        b.make<v0::Convert>(b.make<v0::Squeeze>(attention->input_value(4), batch_and_head), ov::element::i32);

    std::vector<int32_t> slots(static_cast<size_t>(selection_width));
    std::iota(slots.begin(), slots.end(), 0);
    auto slot_positions =
        v0::Constant::create(ov::element::i32, ov::Shape{1, static_cast<size_t>(selection_width)}, slots);
    b.nodes.push_back(slot_positions);
    const auto count_column = b.make<v0::Unsqueeze>(counts, b.axes({1}));
    const auto active_slots = b.make<v1::Less>(slot_positions, count_column);
    const auto safe_indices = b.make<v1::Select>(active_slots, indices, sentinel);

    auto table_shape = v0::Constant::create(ov::element::i64, ov::Shape{2}, {query_length, block_count + 1});
    auto updates_shape = v0::Constant::create(ov::element::i64, ov::Shape{2}, {query_length, selection_width});
    b.nodes.insert(b.nodes.end(), {table_shape, updates_shape});
    const auto table = b.make<v3::Broadcast>(zero, table_shape);
    const auto updates = b.make<v3::Broadcast>(one, updates_shape);
    const auto selected = b.make<v3::ScatterElementsUpdate>(table, safe_indices, updates, b.scalar(1));

    Output<Node> valid_length = total_keys;
    if (attention->get_input_size() >= 6) {
        valid_length =
            b.make<v0::Convert>(b.make<v0::Squeeze>(attention->input_value(5), b.axes({0})), ov::element::i32);
    }
    Output<Node> logical_keys;
    Output<Node> logical_queries;
    Output<Node> valid_keys;
    Output<Node> valid_queries;
    if (physical_attention_mask) {
        const auto mask_shape = physical_attention_mask->get_output_partial_shape(0);
        const PartialShape expected_mask_shape{1, key_length};
        OPENVINO_ASSERT(mask_shape.is_static() && mask_shape == expected_mask_shape,
                        "QSA physical attention mask must have shape [1, key_length], got ",
                        mask_shape);
        const auto physical_i32 = b.make<v0::Convert>(physical_attention_mask, ov::element::i32);
        const auto binary = b.make<v1::NotEqual>(physical_i32, zero);
        const auto binary_i32 = b.make<v0::Convert>(binary, ov::element::i32);
        const auto ranks = b.make<v1::Subtract>(b.make<v0::CumSum>(binary_i32, one), one);
        std::vector<int32_t> query_indices(static_cast<size_t>(query_length));
        std::iota(query_indices.begin(), query_indices.end(), static_cast<int32_t>(key_length - query_length));
        auto gather_indices =
            v0::Constant::create(ov::element::i32, Shape{static_cast<size_t>(query_length)}, query_indices);
        auto transpose_order = b.axes({1, 0});
        b.nodes.push_back(gather_indices);
        logical_keys = ranks;
        logical_queries = b.make<v1::Transpose>(b.make<v8::Gather>(ranks, gather_indices, one), transpose_order);
        valid_keys = binary;
        valid_queries = b.make<v1::Transpose>(b.make<v8::Gather>(binary, gather_indices, one), transpose_order);
    } else {
        std::vector<int32_t> key_positions(static_cast<size_t>(key_length));
        std::iota(key_positions.begin(), key_positions.end(), 0);
        auto positions =
            v0::Constant::create(ov::element::i32, Shape{1, static_cast<size_t>(key_length)}, key_positions);
        b.nodes.push_back(positions);
        const auto padding = b.make<v1::Subtract>(total_keys, valid_length);
        logical_keys = b.make<v1::Subtract>(positions, padding);
        std::vector<int32_t> query_positions(static_cast<size_t>(query_length));
        std::iota(query_positions.begin(), query_positions.end(), 0);
        auto query_offsets =
            v0::Constant::create(ov::element::i32, Shape{static_cast<size_t>(query_length), 1}, query_positions);
        b.nodes.push_back(query_offsets);
        const auto query_base = b.make<v1::Subtract>(valid_length, total_queries);
        logical_queries = b.make<v1::Add>(query_offsets, query_base);
        valid_keys = b.make<v1::LogicalAnd>(b.make<v1::GreaterEqual>(logical_keys, zero),
                                            b.make<v1::Less>(logical_keys, valid_length));
        valid_queries = b.make<v1::GreaterEqual>(logical_queries, zero);
    }
    const auto nonnegative_keys = b.make<v1::Maximum>(logical_keys, zero);
    const auto key_blocks = b.make<v1::Divide>(nonnegative_keys, block);
    const auto bounded_blocks = b.make<v1::Minimum>(key_blocks, sentinel);
    auto query_key_shape = v0::Constant::create(ov::element::i64, ov::Shape{2}, {query_length, key_length});
    b.nodes.push_back(query_key_shape);
    const auto key_block_rows = b.make<v3::Broadcast>(bounded_blocks, query_key_shape);
    const auto selected_tokens = b.make<v1::Greater>(b.make<v6::GatherElements>(selected, key_block_rows, 1), zero);

    const auto complete_blocks = b.make<v1::Divide>(b.make<v1::Add>(logical_queries, one), block);
    const auto tail_start = b.make<v1::Multiply>(complete_blocks, block);
    const auto before_query = b.make<v1::LessEqual>(logical_keys, logical_queries);
    const auto tail = b.make<v1::LogicalAnd>(b.make<v1::GreaterEqual>(logical_keys, tail_start), before_query);
    const auto selected_causal = b.make<v1::LogicalAnd>(selected_tokens, before_query);
    const auto allowed = b.make<v1::LogicalOr>(selected_causal, tail);
    const auto mask = b.make<v1::LogicalAnd>(b.make<v1::LogicalAnd>(allowed, valid_keys), valid_queries);

    auto expand_heads = [&](const Output<Node>& input, const PartialShape& shape) -> Output<Node> {
        if (key_heads == query_heads || key_heads == 1) {
            return input;
        }
        auto expanded = b.make<v0::Unsqueeze>(input, b.axes({2}));
        auto broadcast =
            b.make<v3::Broadcast>(expanded,
                                  b.axes({1, key_heads, query_heads / key_heads, key_length, shape[3].get_length()}));
        return b.make<v1::Reshape>(broadcast, b.axes({1, query_heads, key_length, shape[3].get_length()}), false);
    };
    const auto dense_key = expand_heads(attention->input_value(1), key_shape);
    const auto dense_value = expand_heads(attention->input_value(2), value_shape);
    std::shared_ptr<v13::ScaledDotProductAttention> dense;
    if (attention->get_input_size() >= 7) {
        dense = b.make<v13::ScaledDotProductAttention>(attention->input_value(0),
                                                       dense_key,
                                                       dense_value,
                                                       mask,
                                                       attention->input_value(6),
                                                       false);
    } else {
        dense = b.make<v13::ScaledDotProductAttention>(attention->input_value(0), dense_key, dense_value, mask, false);
    }
    dense->set_friendly_name(attention->get_friendly_name() + "/npuw_dense");
    const auto nonempty_rows = b.make<v1::ReduceLogicalOr>(mask, b.axes({1}), true);
    const auto broadcast_rows = b.make<v0::Unsqueeze>(nonempty_rows, b.axes({0, 1}));
    const auto output_zero = b.make<v1::ConvertLike>(zero, dense);
    const auto output = b.make<v1::Select>(broadcast_rows, dense, output_zero);
    output->set_friendly_name(attention->get_friendly_name());
    ov::copy_runtime_info(attention, b.nodes);
    dense->get_rt_info().erase(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
    output->get_rt_info().erase(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
    ov::replace_node(attention, output);
    return true;
}

bool lower_gathered(const std::shared_ptr<v17::SparseAttention>& attention,
                    const std::shared_ptr<ov::Node>& physical_attention_mask) {
    const auto& query_shape = attention->get_input_partial_shape(0);
    const auto& key_shape = attention->get_input_partial_shape(1);
    const auto& value_shape = attention->get_input_partial_shape(2);
    const auto& indices_shape = attention->get_input_partial_shape(3);
    const auto& counts_shape = attention->get_input_partial_shape(4);
    if (!physical_attention_mask || !attention->get_causal() || attention->get_query_block_size() != 1 ||
        attention->get_input_size() < 6 || attention->get_input_size() > 7 || !query_shape.is_static() ||
        !key_shape.is_static() || !value_shape.is_static() || !indices_shape.is_static() || !counts_shape.is_static() ||
        query_shape.size() != 4 || key_shape.size() != 4 || value_shape.size() != 4 || indices_shape.size() != 4 ||
        counts_shape.size() != 3 || query_shape[0].get_length() != 1 || key_shape[0].get_length() != 1 ||
        value_shape[0].get_length() != 1 || indices_shape[0].get_length() != 1 || indices_shape[1].get_length() != 1 ||
        counts_shape[0].get_length() != 1 || counts_shape[1].get_length() != 1) {
        return false;
    }
    const auto length_node = ov::as_type_ptr<v1::ReduceSum>(attention->get_input_node_shared_ptr(5));
    if (!length_node || length_node->input_value(0) != physical_attention_mask->output(0) ||
        length_node->get_keep_dims()) {
        return false;
    }
    const auto axis_value = ov::as_type_ptr<v0::Constant>(length_node->get_input_node_shared_ptr(1));
    if (!axis_value || ov::shape_size(axis_value->get_shape()) != 1) {
        return false;
    }
    const auto axis = axis_value->cast_vector<int64_t>()[0];
    if (axis != -1 && axis != 1) {
        return false;
    }
    const auto queries = query_shape[2].get_length();
    const auto keys = key_shape[2].get_length();
    const auto heads = query_shape[1].get_length();
    const auto kv_heads = key_shape[1].get_length();
    const auto query_dim = query_shape[3].get_length();
    const auto value_dim = value_shape[3].get_length();
    const auto selection_width = indices_shape[3].get_length();
    const auto block = attention->get_key_block_size();
    if (queries <= 0 || keys <= 0 || queries > keys || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 ||
        query_dim <= 0 || value_dim <= 0 || key_shape[3] != query_shape[3] || value_shape[1] != key_shape[1] ||
        value_shape[2] != key_shape[2] || indices_shape[2] != query_shape[2] || counts_shape[2] != query_shape[2] ||
        selection_width <= 0 || block <= 1 || block > std::numeric_limits<int32_t>::max() ||
        keys > std::numeric_limits<int32_t>::max() - block || queries > std::numeric_limits<int32_t>::max() ||
        selection_width > std::numeric_limits<int32_t>::max() ||
        physical_attention_mask->get_output_partial_shape(0) != (PartialShape{1, keys})) {
        return false;
    }
    const auto block_slots = std::min<int64_t>(selection_width, (keys + block - 1) / block);
    const auto selected_slots = block_slots * block;
    const auto compact_keys = std::min<int64_t>(keys, selected_slots + block - 1);
    Builder b;
    const auto zero = b.scalar(0);
    const auto one = b.scalar(1);
    const auto block_size = b.scalar(static_cast<int32_t>(block));
    const auto mask_i32 = b.make<v0::Convert>(physical_attention_mask, element::i32);
    const auto valid_physical = b.make<v1::NotEqual>(mask_i32, zero);
    const auto mask_values = b.make<v0::Convert>(valid_physical, element::i32);
    const auto cumulative = b.make<v0::CumSum>(mask_values, one);
    const auto logical_ranks = b.make<v1::Subtract>(cumulative, one);
    const auto total_valid = b.make<v0::Squeeze>(b.make<v1::ReduceSum>(mask_values, b.axes({1}), false), b.axes({0}));

    std::vector<int32_t> query_values(static_cast<size_t>(queries));
    std::iota(query_values.begin(), query_values.end(), static_cast<int32_t>(keys - queries));
    auto query_indices = b.indices(Shape{static_cast<size_t>(queries)}, query_values);
    auto query_rank = b.make<v0::Squeeze>(b.make<v8::Gather>(logical_ranks, query_indices, one), b.axes({0}));
    auto query_valid = b.make<v0::Squeeze>(b.make<v8::Gather>(valid_physical, query_indices, one), b.axes({0}));
    auto indices = b.make<v0::Convert>(b.make<v0::Squeeze>(attention->input_value(3), b.axes({0, 1})), element::i32);
    auto limited_indices = b.make<v8::Slice>(indices, b.axes({0}), b.axes({block_slots}), b.axes({1}), b.axes({1}));
    auto counts = b.make<v0::Convert>(b.make<v0::Squeeze>(attention->input_value(4), b.axes({0, 1})), element::i32);
    auto count_column = b.make<v0::Unsqueeze>(counts, b.axes({1}));
    auto selected_count = b.make<v1::Multiply>(count_column, block_size);
    std::vector<int32_t> slot_values(static_cast<size_t>(compact_keys));
    std::iota(slot_values.begin(), slot_values.end(), 0);
    std::vector<int32_t> block_lookup(slot_values.size());
    std::vector<int32_t> block_offsets(slot_values.size());
    for (size_t i = 0; i < slot_values.size(); ++i) {
        block_lookup[i] = static_cast<int32_t>(std::min<int64_t>(static_cast<int64_t>(i) / block, block_slots - 1));
        block_offsets[i] = static_cast<int32_t>(static_cast<int64_t>(i) % block);
    }
    auto slot_range = b.indices(Shape{1, static_cast<size_t>(compact_keys)}, slot_values);
    auto lookup = b.indices(Shape{static_cast<size_t>(compact_keys)}, block_lookup);
    auto offsets = b.indices(Shape{1, static_cast<size_t>(compact_keys)}, block_offsets);
    auto selected_blocks = b.make<v8::Gather>(limited_indices, lookup, one);
    auto selected_positions = b.make<v1::Add>(b.make<v1::Multiply>(selected_blocks, block_size), offsets);
    auto use_selected = b.make<v1::Less>(slot_range, selected_count);
    auto query_column = b.make<v0::Unsqueeze>(query_rank, b.axes({1}));
    auto valid_column = b.make<v0::Unsqueeze>(query_valid, b.axes({1}));
    auto tail_start = b.make<v1::Multiply>(
        b.make<v1::Divide>(b.make<v1::Maximum>(b.make<v1::Add>(query_column, one), zero), block_size),
        block_size);
    auto tail_offsets = b.make<v1::Subtract>(slot_range, selected_count);
    auto tail_positions = b.make<v1::Add>(tail_start, tail_offsets);
    auto logical_positions = b.make<v1::Select>(use_selected, selected_positions, tail_positions);
    auto source_valid = b.make<v1::Select>(use_selected,
                                           b.make<v1::GreaterEqual>(selected_blocks, zero),
                                           b.make<v1::Less>(tail_offsets, b.scalar(static_cast<int32_t>(block - 1))));
    auto active = b.make<v1::LogicalAnd>(
        valid_column,
        b.make<v1::LogicalAnd>(
            source_valid,
            b.make<v1::LogicalAnd>(b.make<v1::GreaterEqual>(logical_positions, zero),
                                   b.make<v1::LogicalAnd>(b.make<v1::LessEqual>(logical_positions, query_column),
                                                          b.make<v1::Less>(logical_positions, total_valid)))));

    // Invert the mask's monotonic prefix counts without relying on ScatterElementsUpdate.
    auto cumulative_1d = b.make<v0::Squeeze>(cumulative, b.axes({0}));
    auto target = b.make<v1::Select>(active, b.make<v1::Add>(logical_positions, one), one);
    Output<Node> low = b.make<v3::Broadcast>(zero, b.axes({queries, compact_keys}));
    Output<Node> high = b.make<v3::Broadcast>(b.scalar(static_cast<int32_t>(keys)), b.axes({queries, compact_keys}));
    for (int64_t remaining = keys; remaining > 0; remaining /= 2) {
        auto middle = b.make<v1::Add>(low, b.make<v1::Divide>(b.make<v1::Subtract>(high, low), b.scalar(2)));
        auto safe_middle = b.make<v1::Minimum>(middle, b.scalar(static_cast<int32_t>(keys - 1)));
        auto before = b.make<v1::Less>(b.make<v8::Gather>(cumulative_1d, safe_middle, zero), target);
        auto next_low = b.make<v1::Select>(before, b.make<v1::Minimum>(b.make<v1::Add>(middle, one), high), low);
        auto next_high = b.make<v1::Select>(before, high, middle);
        low = next_low;
        high = next_high;
    }
    auto physical_positions =
        b.make<v1::Select>(active, b.make<v1::Minimum>(low, b.scalar(static_cast<int32_t>(keys - 1))), zero);
    auto gathered_key = b.make<v8::Gather>(attention->input_value(1), physical_positions, b.scalar(2));
    auto gathered_value = b.make<v8::Gather>(attention->input_value(2), physical_positions, b.scalar(2));

    auto grouped_query = b.make<v1::Reshape>(attention->input_value(0),
                                             b.axes({1, kv_heads, heads / kv_heads, queries, query_dim}),
                                             false);
    auto ordered_query = b.make<v1::Transpose>(grouped_query, b.axes({0, 1, 3, 2, 4}));
    auto scores = b.make<v0::MatMul>(ordered_query, gathered_key, false, true);
    scores->set_friendly_name(attention->get_friendly_name() + "/sparse_scores");
    Output<Node> scale = b.real(1.f / std::sqrt(static_cast<float>(query_dim)));
    if (attention->get_input_size() == 7) {
        scale = attention->input_value(6);
    }
    auto scaled_scores = b.make<v1::Multiply>(scores, b.make<v1::ConvertLike>(scale, scores));
    auto active5 = b.make<v0::Unsqueeze>(active, b.axes({0, 1, 3}));
    auto neg_inf = b.make<v1::ConvertLike>(b.real(-std::numeric_limits<float>::infinity()), scores);
    auto masked_scores = b.make<v1::Select>(active5, scaled_scores, neg_inf);
    auto has_keys = b.make<v1::ReduceLogicalOr>(active, b.axes({1}), true);
    auto has_keys5 = b.make<v0::Unsqueeze>(has_keys, b.axes({0, 1, 3}));
    auto safe_scores = b.make<v1::Select>(has_keys5, masked_scores, b.make<v1::ConvertLike>(zero, scores));
    auto probabilities = b.make<v8::Softmax>(safe_scores, -1);
    auto attended = b.make<v0::MatMul>(probabilities, gathered_value, false, false);
    auto original_order = b.make<v1::Transpose>(attended, b.axes({0, 1, 3, 2, 4}));
    auto output_shape = b.axes({1, heads, queries, value_dim});
    auto reshaped = b.make<v1::Reshape>(original_order, output_shape, false);
    auto has_keys4 = b.make<v0::Unsqueeze>(has_keys, b.axes({0, 1}));
    auto output = b.make<v1::Select>(has_keys4, reshaped, b.make<v1::ConvertLike>(zero, reshaped));
    output->set_friendly_name(attention->get_friendly_name());
    ov::copy_runtime_info(attention, b.nodes);
    output->get_rt_info().erase(ov::npuw::NPUW_SDPA_MASK_RT_KEY);
    ov::replace_node(attention, output);
    return true;
}

}  // namespace

bool ov::npuw::LowerSparseAttention::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto attention = ov::as_type_ptr<v17::SparseAttention>(node)) {
            changed |= lower(attention, m_physical_attention_mask);
        }
    }
    return changed;
}

bool ov::npuw::LowerSparseAttentionToGathered::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto attention = ov::as_type_ptr<v17::SparseAttention>(node)) {
            changed |= lower_gathered(attention, m_physical_attention_mask);
        }
    }
    return changed;
}
