// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qwen_sparse_attention_fusion.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <string>

#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/sparse_attention.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v6 = ov::op::v6;
namespace v8 = ov::op::v8;
namespace v11 = ov::op::v11;
namespace v12 = ov::op::v12;
namespace v13 = ov::op::v13;
namespace v17 = ov::op::v17;

template <typename T>
std::shared_ptr<T> input_node(const std::shared_ptr<Node>& node, size_t index) {
    return node && index < node->get_input_size() ? ov::as_type_ptr<T>(node->get_input_node_shared_ptr(index))
                                                  : nullptr;
}

bool constant_is(const Output<Node>& output, int64_t expected) {
    const auto value = ov::util::get_constant_from_source(output);
    return value && shape_size(value->get_shape()) == 1 && value->cast_vector<int64_t>()[0] == expected;
}

bool constant_vector_is(const Output<Node>& output, const std::vector<int64_t>& expected) {
    const auto value = ov::util::get_constant_from_source(output);
    return value && value->cast_vector<int64_t>() == expected;
}

bool is_zero_one_range(const std::shared_ptr<ov::op::v4::Range>& range, int64_t stop) {
    return range && constant_is(range->input_value(0), 0) && constant_is(range->input_value(1), stop) &&
           constant_is(range->input_value(2), 1);
}

Output<Node> strip_shape_wrappers(Output<Node> value) {
    for (size_t depth = 0; depth < 5; ++depth) {
        const auto node = value.get_node_shared_ptr();
        if (!ov::is_type_any_of<v0::Convert, v0::Unsqueeze, v3::Broadcast>(node)) {
            break;
        }
        value = node->input_value(0);
    }
    return value;
}

bool matches_qsa_mask(const std::shared_ptr<v6::GatherElements>& token_lookup,
                      const std::shared_ptr<v1::Select>& scatter_indices,
                      const std::shared_ptr<v1::LogicalAnd>& tail,
                      const std::shared_ptr<v1::Minimum>& limited_count,
                      const std::shared_ptr<v1::ReduceSum>& key_length,
                      const std::shared_ptr<ov::op::v4::Range>& query_range,
                      int64_t query_size,
                      int64_t key_size,
                      int64_t block_size,
                      int64_t selection_width) {
    auto active_slots = input_node<v1::Less>(scatter_indices, 0);
    auto active_range = input_node<ov::op::v4::Range>(active_slots, 0);
    if (!active_slots || !is_zero_one_range(active_range, selection_width) ||
        strip_shape_wrappers(active_slots->input_value(1)) != limited_count->output(0) ||
        !constant_is(scatter_indices->input_value(2), (key_size - 1) / block_size + 1)) {
        return false;
    }

    auto lookup_broadcast = input_node<v3::Broadcast>(token_lookup, 1);
    auto lookup_expand = input_node<v0::Unsqueeze>(lookup_broadcast, 0);
    auto block_select = input_node<v1::Select>(lookup_expand, 0);
    auto valid_key = input_node<v1::GreaterEqual>(block_select, 0);
    auto block_divide = input_node<v1::Divide>(block_select, 1);
    auto nonnegative_key = input_node<v1::Maximum>(block_divide, 0);
    auto logical_key = input_node<v1::Subtract>(nonnegative_key, 0);
    auto key_row = input_node<v0::Unsqueeze>(logical_key, 0);
    auto key_range = input_node<ov::op::v4::Range>(key_row, 0);
    auto padding_expand = input_node<v0::Unsqueeze>(logical_key, 1);
    auto padding = input_node<v1::Subtract>(padding_expand, 0);
    auto length_broadcast = input_node<v3::Broadcast>(padding, 1);
    if (!lookup_broadcast || !lookup_expand || !constant_vector_is(lookup_expand->input_value(1), {1, 2}) ||
        !block_select || !valid_key || !block_divide || !nonnegative_key || !logical_key || !key_row ||
        !constant_is(key_row->input_value(1), 0) || !is_zero_one_range(key_range, key_size) || !padding_expand ||
        !constant_is(padding_expand->input_value(1), 1) || !padding ||
        !constant_is(padding->input_value(0), key_size) || !length_broadcast ||
        length_broadcast->input_value(0) != key_length->output(0) ||
        valid_key->input_value(0) != logical_key->output(0) || !constant_is(valid_key->input_value(1), 0) ||
        nonnegative_key->input_value(0) != logical_key->output(0) || !constant_is(nonnegative_key->input_value(1), 0) ||
        !constant_is(block_divide->input_value(1), block_size) ||
        !constant_is(block_select->input_value(2), (key_size - 1) / block_size + 2)) {
        return false;
    }

    auto tail_bounds = input_node<v1::LogicalAnd>(tail, 0);
    auto tail_valid = input_node<v1::GreaterEqual>(tail, 1);
    auto tail_lower = input_node<v1::GreaterEqual>(tail_bounds, 0);
    auto tail_upper = input_node<v1::LessEqual>(tail_bounds, 1);
    auto key_expand = input_node<v0::Unsqueeze>(tail_lower, 0);
    auto start_expand = input_node<v0::Unsqueeze>(tail_lower, 1);
    auto query_expand = input_node<v0::Unsqueeze>(tail_upper, 1);
    auto query_position = input_node<v1::Add>(query_expand, 0);
    auto query_past_expand = input_node<v0::Unsqueeze>(query_position, 0);
    auto query_past = input_node<v1::Subtract>(query_past_expand, 0);
    auto query_length_broadcast = input_node<v3::Broadcast>(query_past, 0);
    auto tail_start = input_node<v1::Multiply>(start_expand, 0);
    auto tail_divide = input_node<v1::Divide>(tail_start, 0);
    auto tail_plus_one = input_node<v1::Add>(tail_divide, 0);
    if (!tail_bounds || !tail_valid || !tail_lower || !tail_upper || !key_expand ||
        !constant_is(key_expand->input_value(1), 1) || key_expand->input_value(0) != logical_key->output(0) ||
        tail_valid->input_value(0) != key_expand->output(0) || !constant_is(tail_valid->input_value(1), 0) ||
        tail_upper->input_value(0) != key_expand->output(0) || !start_expand ||
        !constant_is(start_expand->input_value(1), -1) || !query_expand ||
        !constant_is(query_expand->input_value(1), -1) || !query_position ||
        query_position->input_value(1) != query_range->output(0) || !query_past_expand ||
        !constant_is(query_past_expand->input_value(1), 1) || !query_past ||
        !constant_is(query_past->input_value(1), query_size) || !query_length_broadcast ||
        query_length_broadcast->input_value(0) != key_length->output(0) || !tail_start ||
        !constant_is(tail_start->input_value(1), block_size) || !tail_divide || !tail_divide->is_pythondiv() ||
        !constant_is(tail_divide->input_value(1), block_size) || !tail_plus_one ||
        tail_plus_one->input_value(0) != query_position->output(0) || !constant_is(tail_plus_one->input_value(1), 1)) {
        return false;
    }
    return true;
}

struct Match {
    std::shared_ptr<v13::ScaledDotProductAttention> attention;
    std::shared_ptr<v1::Select> selected;
    std::shared_ptr<v0::MatMul> scores;
    Output<Node> block_keys;
    Output<Node> key_length;
    size_t block_size = 0;
    size_t selection_width = 0;
};

std::optional<Match> match(const std::shared_ptr<v13::ScaledDotProductAttention>& attention,
                           const std::shared_ptr<ov::Model>& model) {
    const auto reject = [&](const char* reason) -> std::optional<Match> {
        LOG_DEBUG("QSA fusion skipped " << attention->get_friendly_name() << ": " << reason);
        return std::nullopt;
    };
    if (attention->get_causal() || attention->get_input_size() != 5 ||
        attention->get_input_element_type(3) != element::boolean) {
        return reject("unsupported SDPA inputs");
    }
    const auto query_shape = attention->get_input_partial_shape(0);
    const auto key_shape = attention->get_input_partial_shape(1);
    if (!query_shape.is_static() || !key_shape.is_static() || query_shape.size() != 4 || key_shape.size() != 4 ||
        query_shape[0].get_length() != 1 || key_shape[0].get_length() != 1) {
        return reject("unsupported query or key shape");
    }

    auto reshape = input_node<v1::Reshape>(attention, 3);
    auto broadcast = input_node<v3::Broadcast>(reshape, 0);
    auto expand = input_node<v0::Unsqueeze>(broadcast, 0);
    auto combined = input_node<v1::LogicalOr>(expand, 0);
    if (!reshape || reshape->get_special_zero() || !broadcast || !expand || !constant_is(expand->input_value(1), 2) ||
        !combined) {
        return reject("unrecognized broadcast mask");
    }
    const auto source_shape = combined->get_output_partial_shape(0);
    const auto mask_shape = attention->get_input_partial_shape(3);
    if (!source_shape.is_static() || !mask_shape.is_static() || source_shape.size() != 4 || mask_shape.size() != 4 ||
        source_shape[0].get_length() != 1 || source_shape[1].get_length() != 1 || source_shape[2] != query_shape[2] ||
        source_shape[3] != key_shape[2] || mask_shape[0].get_length() != 1 || mask_shape[1] != query_shape[1] ||
        mask_shape[2] != query_shape[2] || mask_shape[3] != key_shape[2]) {
        return reject("unsupported broadcast mask shape");
    }

    auto selected_mask = input_node<v1::Greater>(combined, 0);
    auto token_lookup = input_node<v6::GatherElements>(selected_mask, 0);
    auto scatter = input_node<v12::ScatterElementsUpdate>(token_lookup, 0);
    auto scatter_indices = input_node<v1::Select>(scatter, 1);
    auto indices_convert = input_node<v0::Convert>(scatter_indices, 1);
    auto selected = input_node<v1::Select>(indices_convert, 0);
    auto padded = input_node<ov::op::util::PadBase>(selected, 1);
    auto topk = input_node<v11::TopK>(padded, 0);
    auto tail_expand = input_node<v0::Unsqueeze>(combined, 1);
    auto tail = input_node<v1::LogicalAnd>(tail_expand, 0);
    auto scatter_data = input_node<v3::Broadcast>(scatter, 0);
    auto scatter_updates = input_node<v3::Broadcast>(scatter, 2);
    if (!selected_mask || !constant_is(selected_mask->input_value(1), 0) || !token_lookup ||
        token_lookup->get_axis() != 3 || !scatter || !constant_is(scatter->input_value(3), -1) ||
        scatter->get_reduction() != v12::ScatterElementsUpdate::Reduction::NONE || !scatter->get_use_init_val() ||
        !scatter_data || !constant_is(scatter_data->input_value(0), 0) || !scatter_updates ||
        !constant_is(scatter_updates->input_value(0), 1) || !scatter_indices || !indices_convert || !selected ||
        !constant_is(selected->input_value(2), -1) || !padded || padded->get_input_size() != 4 ||
        padded->get_pad_mode() != ov::op::PadMode::CONSTANT ||
        !constant_vector_is(padded->input_value(1), {0, 0, 0, 0}) || !constant_is(padded->input_value(3), -1) ||
        !topk || padded->input_value(0) != topk->output(1) || topk->get_axis() != 3 ||
        topk->get_mode() != ov::op::TopKMode::MAX || topk->get_sort_type() != ov::op::TopKSortType::SORT_VALUES ||
        !topk->get_stable() || !tail_expand || !constant_is(tail_expand->input_value(1), 1) || !tail) {
        return reject("unrecognized TopK selection or causal tail");
    }
    const auto selection_shape = selected->get_output_partial_shape(0);
    if (!selection_shape.is_static() || selection_shape.size() != 4 || selection_shape[0].get_length() != 1 ||
        selection_shape[1].get_length() != 1 || selection_shape[2] != query_shape[2] ||
        selection_shape[3].get_length() <= 0 || selected->get_input_element_type(0) != element::boolean) {
        return reject("unsupported selection shape");
    }
    const auto selection_width = static_cast<size_t>(selection_shape[3].get_length());

    auto score_mask = input_node<v1::Select>(topk, 0);
    auto reduced = input_node<v1::ReduceSum>(score_mask, 1);
    auto score_reshape = input_node<v1::Reshape>(reduced, 0);
    auto relu = input_node<v0::Relu>(score_reshape, 0);
    auto matmul = input_node<v0::MatMul>(relu, 0);
    auto key_expand = input_node<v0::Unsqueeze>(matmul, 1);
    auto key_concat = input_node<v0::Concat>(key_expand, 0);
    auto sentinel_key = input_node<v3::Broadcast>(key_concat, 1);
    if (!score_mask || !reduced || !constant_is(reduced->input_value(1), 2) || !score_reshape || !relu || !matmul ||
        matmul->get_transpose_a() || !matmul->get_transpose_b() || !key_expand ||
        !constant_is(key_expand->input_value(1), 1) || !key_concat || key_concat->get_axis() != 1 ||
        key_concat->get_input_size() != 2 || !sentinel_key || !constant_is(sentinel_key->input_value(0), 0)) {
        return reject("unrecognized index score computation");
    }
    const auto fill = ov::util::get_constant_from_source(score_mask->input_value(2));
    if (!fill || shape_size(fill->get_shape()) != 1 || !std::isinf(fill->cast_vector<float>()[0]) ||
        fill->cast_vector<float>()[0] >= 0.0f) {
        return reject("index score mask does not use negative infinity");
    }
    auto block_slice = input_node<v8::Slice>(key_concat, 0);
    auto cache_scatter = input_node<v12::ScatterElementsUpdate>(block_slice, 0);
    auto cache_select = input_node<v1::Select>(cache_scatter, 0);
    auto cache_concat = input_node<v0::Concat>(cache_select, 1);
    auto block_parameter = input_node<v0::Parameter>(cache_concat, 0);
    if (!block_slice || !cache_scatter || !cache_select || !cache_concat || !block_parameter ||
        cache_concat->get_axis() != 1) {
        return reject("unrecognized block-key cache");
    }
    std::string block_name;
    for (const auto& name : block_parameter->output(0).get_names()) {
        if (ov::npuw::util::isQSABlockIndexCache(name)) {
            block_name = name;
            break;
        }
    }
    if (block_name.empty()) {
        return reject("block-key input is not a QSA cache");
    }
    const auto block_size = ov::npuw::util::qsaIndexBlockSize(model, block_name);
    if (selection_width > static_cast<size_t>(std::numeric_limits<int64_t>::max()) / block_size) {
        return reject("selection budget is too large");
    }

    auto valid_expand = input_node<v0::Unsqueeze>(selected, 0);
    auto valid_slots = input_node<v1::Less>(valid_expand, 0);
    auto count_expand = input_node<v0::Unsqueeze>(valid_slots, 1);
    auto limited_count = input_node<v1::Minimum>(count_expand, 0);
    auto visible = input_node<v1::Maximum>(limited_count, 0);
    auto visible_divide = input_node<v1::Divide>(visible, 0);
    auto plus_one = input_node<v1::Add>(visible_divide, 0);
    auto query_positions = input_node<v1::Add>(plus_one, 0);
    auto query_range = input_node<ov::op::v4::Range>(query_positions, 1);
    auto past_expand = input_node<v0::Unsqueeze>(query_positions, 0);
    auto past_broadcast = input_node<v3::Broadcast>(past_expand, 0);
    auto past_length = input_node<v1::Subtract>(past_broadcast, 0);
    auto sum = input_node<v1::ReduceSum>(past_length, 0);
    auto mask_parameter = input_node<v0::Parameter>(sum, 0);
    if (!valid_expand || !constant_is(valid_expand->input_value(1), 1) || !valid_slots || !count_expand ||
        !constant_is(count_expand->input_value(1), -1) || !limited_count ||
        !constant_is(limited_count->input_value(1), static_cast<int64_t>(selection_width)) || !visible ||
        !constant_is(visible->input_value(1), 0) || !visible_divide ||
        !constant_is(visible_divide->input_value(1), static_cast<int64_t>(block_size)) || !plus_one ||
        !constant_is(plus_one->input_value(1), 1) || !query_positions || !query_range ||
        !constant_is(query_range->input_value(0), 0) ||
        !constant_is(query_range->input_value(1), query_shape[2].get_length()) ||
        !constant_is(query_range->input_value(2), 1) || !past_expand || !constant_is(past_expand->input_value(1), 1) ||
        !past_broadcast || !past_length || !sum ||
        !constant_is(past_length->input_value(1), query_shape[2].get_length()) || sum->get_keep_dims() ||
        !constant_is(sum->input_value(1), -1) || !mask_parameter ||
        mask_parameter->output(0).get_names().count("attention_mask") == 0) {
        return reject("unrecognized per-query selection count");
    }
    std::vector<int64_t> expected_slots(selection_width);
    std::iota(expected_slots.begin(), expected_slots.end(), 0);
    if (!constant_vector_is(valid_slots->input_value(0), expected_slots) ||
        mask_parameter->get_output_partial_shape(0) != PartialShape{1, key_shape[2]}) {
        return reject("selection slots or attention-mask shape differ from QSA");
    }
    auto score_expand = input_node<v0::Unsqueeze>(score_mask, 0);
    auto score_visible = input_node<v1::Less>(score_expand, 0);
    auto score_range = input_node<ov::op::v4::Range>(score_visible, 0);
    auto visible_expand = input_node<v0::Unsqueeze>(score_visible, 1);
    if (!score_expand || !constant_is(score_expand->input_value(1), 1) || !score_visible || !visible_expand ||
        !constant_is(visible_expand->input_value(1), -1) || visible_expand->input_value(0) != visible->output(0) ||
        !score_range || !constant_is(score_range->input_value(0), 0) || !constant_is(score_range->input_value(2), 1)) {
        return reject("index scores and selection counts use different visibility");
    }
    const auto block_shape = block_slice->get_output_partial_shape(0);
    const auto index_query_shape = matmul->get_input_partial_shape(0);
    if (!block_shape.is_static() || block_shape.size() != 3 || !index_query_shape.is_static() ||
        index_query_shape.size() != 4 || index_query_shape[0].get_length() != 1 ||
        index_query_shape[2] != query_shape[2] || index_query_shape[3] != block_shape[2] ||
        !is_zero_one_range(score_range, block_shape[1].get_length() + 1) ||
        !constant_is(topk->input_value(1),
                     std::min<int64_t>(static_cast<int64_t>(selection_width), block_shape[1].get_length() + 1)) ||
        !matches_qsa_mask(token_lookup,
                          scatter_indices,
                          tail,
                          limited_count,
                          sum,
                          query_range,
                          query_shape[2].get_length(),
                          key_shape[2].get_length(),
                          static_cast<int64_t>(block_size),
                          static_cast<int64_t>(selection_width))) {
        return reject("QSA mask does not match the selected blocks and causal tail");
    }
    return Match{attention, selected, matmul, block_slice->output(0), sum->output(0), block_size, selection_width};
}

}  // namespace

bool ov::npuw::QwenSparseAttentionFusion::run_on_model(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(m_indexers.empty(), "QwenSparseAttentionFusion cannot be reused before lowering its indexers.");
    for (const auto& node : model->get_ordered_ops()) {
        const auto attention = ov::as_type_ptr<v13::ScaledDotProductAttention>(node);
        if (!attention) {
            continue;
        }
        const auto matched = match(attention, model);
        if (!matched) {
            continue;
        }
        auto indexer = std::make_shared<v17::SparseAttentionIndexer>(
            matched->scores->input_value(0),
            matched->block_keys,
            matched->key_length,
            static_cast<int64_t>(matched->block_size),
            static_cast<int64_t>(matched->block_size * matched->selection_width),
            1);
        auto counts_i32 = std::make_shared<v0::Convert>(matched->selected->input_value(0), element::i32);
        auto count_axis = v0::Constant::create(element::i64, Shape{1}, {-1});
        auto counts = std::make_shared<v1::ReduceSum>(counts_i32, count_axis, false);
        auto sparse = std::make_shared<v17::SparseAttention>(attention->input_value(0),
                                                             attention->input_value(1),
                                                             attention->input_value(2),
                                                             indexer->output(0),
                                                             indexer->output(1),
                                                             matched->key_length,
                                                             attention->input_value(4),
                                                             static_cast<int64_t>(matched->block_size),
                                                             1,
                                                             true);
        sparse->set_friendly_name(attention->get_friendly_name());
        ov::copy_runtime_info(attention, {indexer, counts_i32, count_axis, counts, sparse});
        m_indexers.push_back({indexer, matched->selected->output(0), counts->output(0)});
        ov::replace_node(attention, sparse);
    }
    return !m_indexers.empty();
}

void ov::npuw::QwenSparseAttentionFusion::lower_indexers() {
    for (const auto& indexer : m_indexers) {
        ov::replace_node(indexer.node, OutputVector{indexer.indices, indexer.counts});
    }
    m_indexers.clear();
}
