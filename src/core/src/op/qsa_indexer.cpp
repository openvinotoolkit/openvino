// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"

#include <cmath>
#include <initializer_list>
#include <string>
#include <utility>

#include "itt.hpp"

namespace ov::op::internal {

namespace {
// The kernels read every integer input as int32_t.
void check_i32_input(const ov::Node* node, size_t idx, const char* name, int64_t rank) {
    const auto& et = node->get_input_element_type(idx);
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node, et.is_dynamic() || et == element::i32, node->get_type_name(), " expects an i32 ", name, " input, got ", et);
    NODE_VALIDATION_CHECK(node, ps.rank().compatible(rank), node->get_type_name(), " expects a rank-", rank, " ", name, " input, got ", ps);
}

void check_shape(const ov::Node* node, size_t idx, const char* name, const ov::PartialShape& expected) {
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node, ps.compatible(expected), node->get_type_name(), " expects ", name, " shaped ", expected, ", got ", ps);
}

void check_float_inputs(const ov::Node* node, std::initializer_list<size_t> indices) {
    element::Type et = element::dynamic;
    for (auto idx : indices) {
        const auto& in_et = node->get_input_element_type(idx);
        NODE_VALIDATION_CHECK(node,
                              element::Type::merge(et, et, in_et),
                              node->get_type_name(),
                              " expects all floating-point inputs to share one element type; input ",
                              idx,
                              " is ",
                              in_et);
    }
    NODE_VALIDATION_CHECK(node, et.is_dynamic() || et.is_real(), node->get_type_name(), " expects floating-point inputs, got ", et);
}

ov::Dimension dim_at(const ov::PartialShape& ps, size_t axis) {
    return ps.rank().is_static() && ps.size() > axis ? ps[axis] : ov::Dimension::dynamic();
}
}  // namespace

QSAIndexer::QSAIndexer(const ov::OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

bool QSAIndexer::validate_config(const Config& c, std::string& reason) {
    auto fail = [&reason](const char* msg) {
        reason = msg;
        return false;
    };
    if (c.key_block_size == 0) {
        return fail("key_block_size must be positive");
    }
    if (c.block_topk == 0) {
        return fail("block_topk must be positive");
    }
    if (c.num_selection_heads == 0) {
        return fail("num_selection_heads must be positive");
    }
    if (c.indexer_head_dim == 0) {
        return fail("indexer_head_dim must be positive");
    }
    if (c.indexer_rotary_dim % 2 != 0 || c.indexer_rotary_dim > c.indexer_head_dim) {
        return fail("indexer_rotary_dim must be even and not exceed indexer_head_dim");
    }
    if (!std::isfinite(c.scale) || c.scale < 0.0f) {
        return fail("scale must be positive (0 selects the default 1 / sqrt(indexer_head_dim))");
    }
    return true;
}

float QSAIndexer::effective_scale(const Config& c) {
    return c.scale > 0.0f ? c.scale : 1.0f / std::sqrt(static_cast<float>(c.indexer_head_dim));
}

size_t QSAIndexer::get_index_query_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::INDEX_QUERY);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[2].is_static(), "QSAIndexer: index_query shape is not static");
    return static_cast<size_t>(ps[2].get_length());
}

size_t QSAIndexer::get_index_key_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::RAW_INDEX_KEYS);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[2].is_static(), "QSAIndexer: raw_index_keys shape is not static");
    return static_cast<size_t>(ps[2].get_length());
}

bool QSAIndexer::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_QSAIndexer_visit_attributes);
    visitor.on_attribute("key_block_size", m_config.key_block_size);
    visitor.on_attribute("block_topk", m_config.block_topk);
    visitor.on_attribute("num_selection_heads", m_config.num_selection_heads);
    visitor.on_attribute("indexer_rotary_dim", m_config.indexer_rotary_dim);
    visitor.on_attribute("indexer_head_dim", m_config.indexer_head_dim);
    visitor.on_attribute("epsilon", m_config.epsilon);
    visitor.on_attribute("scale", m_config.scale);
    return true;
}

void QSAIndexer::validate_and_infer_types() {
    OV_OP_SCOPE(internal_QSAIndexer_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == InputIdx::COUNT, "QSAIndexer expects ", InputIdx::COUNT, " inputs, got ", get_input_size());

    std::string reason;
    NODE_VALIDATION_CHECK(this, validate_config(m_config, reason), "QSAIndexer configuration is invalid: ", reason);

    const auto& index_query_ps = get_input_partial_shape(InputIdx::INDEX_QUERY);
    const auto& raw_index_keys_ps = get_input_partial_shape(InputIdx::RAW_INDEX_KEYS);
    NODE_VALIDATION_CHECK(this,
                           index_query_ps.rank().is_dynamic() || index_query_ps.size() == 4,
                           "QSAIndexer expects a rank-4 [B, L, H_iq, D_idx] index_query input, got ",
                           index_query_ps);
    NODE_VALIDATION_CHECK(this,
                           raw_index_keys_ps.rank().is_dynamic() || raw_index_keys_ps.size() == 4,
                           "QSAIndexer expects a rank-4 [B, L, H_ik, D_idx] raw_index_keys input, got ",
                           raw_index_keys_ps);
    check_float_inputs(this,
                       {InputIdx::INDEX_QUERY,
                        InputIdx::RAW_INDEX_KEYS,
                        InputIdx::ROPE_COS,
                        InputIdx::ROPE_SIN,
                        InputIdx::Q_NORM_WEIGHT,
                        InputIdx::K_NORM_WEIGHT,
                        InputIdx::BLOCK_INDEX_KEYS_CACHE,
                        InputIdx::PAST_INCOMPLETE_RAW_K,
                        InputIdx::PAST_INCOMPLETE_ROPE});

    const auto r = static_cast<int64_t>(m_config.key_block_size);
    const ov::Dimension di(static_cast<int64_t>(m_config.indexer_head_dim));
    const ov::Dimension rot(static_cast<int64_t>(m_config.indexer_rotary_dim));
    const ov::Dimension rope_row(static_cast<int64_t>(2 * m_config.indexer_rotary_dim));
    const auto batch = dim_at(index_query_ps, 0);
    const auto length = dim_at(index_query_ps, 1);
    const auto h_ik = dim_at(raw_index_keys_ps, 2);

    check_shape(this, InputIdx::INDEX_QUERY, "index_query", ov::PartialShape{batch, length, -1, di});
    check_shape(this, InputIdx::RAW_INDEX_KEYS, "raw_index_keys", ov::PartialShape{batch, length, h_ik, di});
    check_shape(this, InputIdx::ROPE_COS, "rope_cos", ov::PartialShape{batch, length, rot});
    check_shape(this, InputIdx::ROPE_SIN, "rope_sin", ov::PartialShape{batch, length, rot});
    check_shape(this, InputIdx::Q_NORM_WEIGHT, "q_norm_weight", ov::PartialShape{di});
    check_shape(this, InputIdx::K_NORM_WEIGHT, "k_norm_weight", ov::PartialShape{di});
    check_shape(this, InputIdx::BLOCK_INDEX_KEYS_CACHE, "block_index_keys_cache", ov::PartialShape{batch, h_ik, -1, di});
    check_shape(this, InputIdx::PAST_INCOMPLETE_RAW_K, "past_incomplete_raw_k", ov::PartialShape{batch, h_ik, r - 1, di});
    check_shape(this, InputIdx::PAST_INCOMPLETE_ROPE, "past_incomplete_rope", ov::PartialShape{batch, rope_row});

    check_i32_input(this, InputIdx::KEY_LENGTH, "key_length", 1);
    check_shape(this, InputIdx::KEY_LENGTH, "key_length", ov::PartialShape{batch});

    if (index_query_ps.rank().is_static() && index_query_ps.size() == 4 && index_query_ps[2].is_static() && h_ik.is_static()) {
        const auto h_iq = static_cast<size_t>(index_query_ps[2].get_length());
        const auto h_ik_value = static_cast<size_t>(h_ik.get_length());
        NODE_VALIDATION_CHECK(this, h_ik_value != 0 && h_iq % h_ik_value == 0, "QSAIndexer: H_iq must be a positive multiple of H_ik.");
        NODE_VALIDATION_CHECK(this, m_config.num_selection_heads == 1 || m_config.num_selection_heads == h_ik_value, "QSAIndexer: num_selection_heads must be 1 or H_ik.");
    }

    const ov::PartialShape selected_block_indices_ps{batch, ov::Dimension(static_cast<int64_t>(m_config.num_selection_heads)), length,
                                           ov::Dimension(static_cast<int64_t>(m_config.block_topk))};
    const ov::PartialShape selected_block_count_ps{batch, ov::Dimension(static_cast<int64_t>(m_config.num_selection_heads)), length};

    set_output_size(OutputIdx::COUNT);
    set_output_type(OutputIdx::SELECTED_BLOCK_INDICES, element::i32, selected_block_indices_ps);
    set_output_type(OutputIdx::SELECTED_BLOCK_COUNT, element::i32, selected_block_count_ps);
    for (auto [out, in] : {std::pair{OutputIdx::BLOCK_INDEX_KEYS_CACHE, InputIdx::BLOCK_INDEX_KEYS_CACHE},
                           std::pair{OutputIdx::PAST_INCOMPLETE_RAW_K, InputIdx::PAST_INCOMPLETE_RAW_K},
                           std::pair{OutputIdx::PAST_INCOMPLETE_ROPE, InputIdx::PAST_INCOMPLETE_ROPE}}) {
        set_output_type(out, get_input_element_type(in), get_input_partial_shape(in));
    }
}

std::shared_ptr<Node> QSAIndexer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<QSAIndexer>(new_args, m_config);
}

}  // namespace ov::op::internal
