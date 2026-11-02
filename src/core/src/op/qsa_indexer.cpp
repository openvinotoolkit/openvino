// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"

#include <cmath>

#include "itt.hpp"

namespace ov::op::internal {
namespace {

Dimension dim_at(const PartialShape& shape, size_t axis) {
    return shape.rank().is_static() && shape.size() > axis ? shape[axis] : Dimension::dynamic();
}

void check_shape(const Node* node, size_t idx, const char* name, const PartialShape& expected) {
    const auto& shape = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node,
                          shape.compatible(expected),
                          "QSAIndexer expects ",
                          name,
                          " shaped ",
                          expected,
                          ", got ",
                          shape);
}

}  // namespace

QSAIndexer::QSAIndexer(const ov::OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

bool QSAIndexer::validate_config(const Config& config, std::string& reason) {
    const auto fail = [&reason](const char* message) {
        reason = message;
        return false;
    };
    if (config.key_block_size == 0) {
        return fail("key_block_size must be positive");
    }
    if (config.block_topk == 0) {
        return fail("block_topk must be positive");
    }
    if (config.num_selection_heads == 0) {
        return fail("num_selection_heads must be positive");
    }
    if (config.indexer_head_dim == 0) {
        return fail("indexer_head_dim must be positive");
    }
    if (config.indexer_rotary_dim % 2 != 0 || config.indexer_rotary_dim > config.indexer_head_dim) {
        return fail("indexer_rotary_dim must be even and not exceed indexer_head_dim");
    }
    if (!std::isfinite(config.scale) || config.scale < 0.0f) {
        return fail("scale must be positive (0 selects the default 1 / sqrt(indexer_head_dim))");
    }
    if (!std::isfinite(config.epsilon) || config.epsilon <= 0.0f) {
        return fail("epsilon must be positive");
    }
    return true;
}

float QSAIndexer::effective_scale(const Config& config) {
    return config.scale > 0.0f ? config.scale : 1.0f / std::sqrt(static_cast<float>(config.indexer_head_dim));
}

bool QSAIndexer::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(QSAIndexer_visit_attributes);
    visitor.on_attribute("key_block_size", m_config.key_block_size);
    visitor.on_attribute("block_topk", m_config.block_topk);
    visitor.on_attribute("num_selection_heads", m_config.num_selection_heads);
    visitor.on_attribute("indexer_rotary_dim", m_config.indexer_rotary_dim);
    visitor.on_attribute("indexer_head_dim", m_config.indexer_head_dim);
    visitor.on_attribute("epsilon", m_config.epsilon);
    visitor.on_attribute("scale", m_config.scale);
    visitor.on_attribute("query_norm", m_config.query_norm);
    visitor.on_attribute("query_rope", m_config.query_rope);
    return true;
}

void QSAIndexer::validate_and_infer_types() {
    OV_OP_SCOPE(QSAIndexer_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == InputIdx::COUNT,
                          "QSAIndexer expects ",
                          InputIdx::COUNT,
                          " inputs, got ",
                          get_input_size());
    std::string reason;
    NODE_VALIDATION_CHECK(this, validate_config(m_config, reason), "QSAIndexer configuration is invalid: ", reason);

    auto data_type = element::dynamic;
    for (const auto idx : {InputIdx::INDEX_QUERY,
                           InputIdx::RAW_INDEX_KEYS,
                           InputIdx::ROPE_COS,
                           InputIdx::ROPE_SIN,
                           InputIdx::Q_NORM_WEIGHT,
                           InputIdx::K_NORM_WEIGHT,
                           InputIdx::BLOCK_INDEX_KEYS_CACHE,
                           InputIdx::PAST_INCOMPLETE_RAW_K,
                           InputIdx::PAST_INCOMPLETE_ROPE}) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(data_type, data_type, get_input_element_type(idx)),
                              "QSAIndexer expects all floating-point inputs to share one element type; input ",
                              idx,
                              " is ",
                              get_input_element_type(idx));
    }
    NODE_VALIDATION_CHECK(this,
                          data_type.is_dynamic() || data_type.is_real(),
                          "QSAIndexer expects floating-point inputs, got ",
                          data_type);
    const auto& length_type = get_input_element_type(InputIdx::KEY_LENGTH);
    NODE_VALIDATION_CHECK(this,
                          length_type.is_dynamic() || length_type == element::i32,
                          "QSAIndexer expects an i32 key_length input, got ",
                          length_type);

    const auto& query_shape = get_input_partial_shape(InputIdx::INDEX_QUERY);
    const auto& raw_keys_shape = get_input_partial_shape(InputIdx::RAW_INDEX_KEYS);
    NODE_VALIDATION_CHECK(this,
                          query_shape.rank().compatible(4),
                          "QSAIndexer expects a rank-4 [B, L, H_iq, D_idx] index_query input, got ",
                          query_shape);
    NODE_VALIDATION_CHECK(this,
                          raw_keys_shape.rank().compatible(4),
                          "QSAIndexer expects a rank-4 [B, L, H_ik, D_idx] raw_index_keys input, got ",
                          raw_keys_shape);

    const auto r = static_cast<int64_t>(m_config.key_block_size);
    const Dimension head_size(static_cast<int64_t>(m_config.indexer_head_dim));
    const Dimension rotary_dim(static_cast<int64_t>(m_config.indexer_rotary_dim));
    const Dimension rope_row(static_cast<int64_t>(2 * m_config.indexer_rotary_dim));
    const auto batch = dim_at(query_shape, 0);
    const auto length = dim_at(query_shape, 1);
    const auto key_heads = dim_at(raw_keys_shape, 2);

    check_shape(this, InputIdx::INDEX_QUERY, "index_query", PartialShape{batch, length, -1, head_size});
    check_shape(this, InputIdx::RAW_INDEX_KEYS, "raw_index_keys", PartialShape{batch, length, key_heads, head_size});
    check_shape(this, InputIdx::ROPE_COS, "rope_cos", PartialShape{batch, length, rotary_dim});
    check_shape(this, InputIdx::ROPE_SIN, "rope_sin", PartialShape{batch, length, rotary_dim});
    check_shape(this, InputIdx::KEY_LENGTH, "key_length", PartialShape{batch});
    check_shape(this, InputIdx::Q_NORM_WEIGHT, "q_norm_weight", PartialShape{head_size});
    check_shape(this, InputIdx::K_NORM_WEIGHT, "k_norm_weight", PartialShape{head_size});
    check_shape(this,
                InputIdx::BLOCK_INDEX_KEYS_CACHE,
                "block_index_keys_cache",
                PartialShape{batch, key_heads, -1, head_size});
    check_shape(this,
                InputIdx::PAST_INCOMPLETE_RAW_K,
                "past_incomplete_raw_k",
                PartialShape{batch, key_heads, r - 1, head_size});
    check_shape(this, InputIdx::PAST_INCOMPLETE_ROPE, "past_incomplete_rope", PartialShape{batch, rope_row});

    const auto query_heads = dim_at(query_shape, 2);
    if (query_heads.is_static() && key_heads.is_static()) {
        const auto h_iq = static_cast<size_t>(query_heads.get_length());
        const auto h_ik = static_cast<size_t>(key_heads.get_length());
        NODE_VALIDATION_CHECK(this,
                              h_ik != 0 && h_iq % h_ik == 0,
                              "QSAIndexer: H_iq must be a positive multiple of H_ik.");
        NODE_VALIDATION_CHECK(this,
                              m_config.num_selection_heads == 1 || m_config.num_selection_heads == h_ik,
                              "QSAIndexer: num_selection_heads must be 1 or H_ik.");
    }

    const Dimension selections(static_cast<int64_t>(m_config.num_selection_heads));
    set_output_size(OutputIdx::COUNT);
    set_output_type(OutputIdx::SELECTED_BLOCK_INDICES,
                    element::i32,
                    PartialShape{batch, selections, length, Dimension(static_cast<int64_t>(m_config.block_topk))});
    set_output_type(OutputIdx::SELECTED_BLOCK_COUNT, element::i32, PartialShape{batch, selections, length});
    for (const auto& [out, in] : {std::pair{OutputIdx::BLOCK_INDEX_KEYS_CACHE, InputIdx::BLOCK_INDEX_KEYS_CACHE},
                                  std::pair{OutputIdx::PAST_INCOMPLETE_RAW_K, InputIdx::PAST_INCOMPLETE_RAW_K},
                                  std::pair{OutputIdx::PAST_INCOMPLETE_ROPE, InputIdx::PAST_INCOMPLETE_ROPE}}) {
        set_output_type(out, get_input_element_type(in), get_input_partial_shape(in));
    }
}

std::shared_ptr<Node> QSAIndexer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(QSAIndexer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<QSAIndexer>(new_args, m_config);
}

}  // namespace ov::op::internal
