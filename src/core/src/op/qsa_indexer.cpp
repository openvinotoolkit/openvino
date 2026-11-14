// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>

#include "itt.hpp"
#include "qsa_indexer_shape_inference.hpp"

namespace ov::op::internal {

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
    const auto max_dimension = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    if (c.key_block_size > max_dimension || c.block_topk > max_dimension || c.num_selection_heads > max_dimension ||
        c.indexer_head_dim > max_dimension || c.indexer_rotary_dim > max_dimension / 2) {
        return fail("QSAIndexer dimensions must fit in int64_t");
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
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[2].is_static(),
                    "QSAIndexer: index_query shape is not static");
    return static_cast<size_t>(ps[2].get_length());
}

size_t QSAIndexer::get_index_key_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::RAW_INDEX_KEYS);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[2].is_static(),
                    "QSAIndexer: raw_index_keys shape is not static");
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
    visitor.on_attribute("query_norm", m_config.query_norm);
    visitor.on_attribute("query_rope", m_config.query_rope);
    return true;
}

void QSAIndexer::validate_and_infer_types() {
    OV_OP_SCOPE(internal_QSAIndexer_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == InputIdx::COUNT,
                          "QSAIndexer expects ",
                          InputIdx::COUNT,
                          " inputs, got ",
                          get_input_size());

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
    // All floating-point inputs must share one element type.
    element::Type float_et = element::dynamic;
    for (const auto idx : {InputIdx::INDEX_QUERY,
                           InputIdx::RAW_INDEX_KEYS,
                           InputIdx::ROPE_COS,
                           InputIdx::ROPE_SIN,
                           InputIdx::Q_NORM_WEIGHT,
                           InputIdx::K_NORM_WEIGHT,
                           InputIdx::BLOCK_INDEX_KEYS_CACHE,
                           InputIdx::PAST_INCOMPLETE_RAW_K,
                           InputIdx::PAST_INCOMPLETE_ROPE}) {
        const auto& in_et = get_input_element_type(idx);
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(float_et, float_et, in_et),
                              "QSAIndexer expects all floating-point inputs to share one element type; input ",
                              idx,
                              " is ",
                              in_et);
    }
    NODE_VALIDATION_CHECK(this,
                          float_et.is_dynamic() || float_et.is_real(),
                          "QSAIndexer expects floating-point inputs, got ",
                          float_et);

    const auto r = static_cast<int64_t>(m_config.key_block_size);
    const ov::Dimension di(static_cast<int64_t>(m_config.indexer_head_dim));
    const ov::Dimension rot(static_cast<int64_t>(m_config.indexer_rotary_dim));
    const ov::Dimension rope_row(static_cast<int64_t>(2 * m_config.indexer_rotary_dim));
    const auto batch = index_query_ps.rank().is_static() ? index_query_ps[0] : ov::Dimension::dynamic();
    const auto length = index_query_ps.rank().is_static() ? index_query_ps[1] : ov::Dimension::dynamic();
    const auto h_iq = index_query_ps.rank().is_static() ? index_query_ps[2] : ov::Dimension::dynamic();
    const auto h_ik = raw_index_keys_ps.rank().is_static() ? raw_index_keys_ps[2] : ov::Dimension::dynamic();

    NODE_VALIDATION_CHECK(this,
                          index_query_ps.compatible(ov::PartialShape{batch, length, -1, di}),
                          "QSAIndexer expects index_query shaped [B, L, H_iq, indexer_head_dim], got ",
                          index_query_ps);
    NODE_VALIDATION_CHECK(this,
                          raw_index_keys_ps.compatible(ov::PartialShape{batch, length, h_ik, di}),
                          "QSAIndexer expects raw_index_keys shaped [B, L, H_ik, indexer_head_dim], got ",
                          raw_index_keys_ps);
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::ROPE_COS).compatible(ov::PartialShape{batch, length, rot}),
                          "QSAIndexer expects rope_cos shaped [B, L, indexer_rotary_dim], got ",
                          get_input_partial_shape(InputIdx::ROPE_COS));
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::ROPE_SIN).compatible(ov::PartialShape{batch, length, rot}),
                          "QSAIndexer expects rope_sin shaped [B, L, indexer_rotary_dim], got ",
                          get_input_partial_shape(InputIdx::ROPE_SIN));
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::Q_NORM_WEIGHT).compatible(ov::PartialShape{di}),
                          "QSAIndexer expects q_norm_weight shaped [indexer_head_dim], got ",
                          get_input_partial_shape(InputIdx::Q_NORM_WEIGHT));
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::K_NORM_WEIGHT).compatible(ov::PartialShape{di}),
                          "QSAIndexer expects k_norm_weight shaped [indexer_head_dim], got ",
                          get_input_partial_shape(InputIdx::K_NORM_WEIGHT));
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(InputIdx::BLOCK_INDEX_KEYS_CACHE).compatible(ov::PartialShape{batch, h_ik, -1, di}),
        "QSAIndexer expects block_index_keys_cache shaped [B, H_ik, blocks, indexer_head_dim], got ",
        get_input_partial_shape(InputIdx::BLOCK_INDEX_KEYS_CACHE));
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(InputIdx::PAST_INCOMPLETE_RAW_K).compatible(ov::PartialShape{batch, h_ik, r - 1, di}),
        "QSAIndexer expects past_incomplete_raw_k shaped "
        "[B, H_ik, key_block_size-1, indexer_head_dim], got ",
        get_input_partial_shape(InputIdx::PAST_INCOMPLETE_RAW_K));
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(InputIdx::PAST_INCOMPLETE_ROPE).compatible(ov::PartialShape{batch, rope_row}),
        "QSAIndexer expects past_incomplete_rope shaped [B, 2*indexer_rotary_dim], got ",
        get_input_partial_shape(InputIdx::PAST_INCOMPLETE_ROPE));

    // The kernels read key_length as int32_t.
    const auto& key_length_et = get_input_element_type(InputIdx::KEY_LENGTH);
    NODE_VALIDATION_CHECK(this,
                          key_length_et.is_dynamic() || key_length_et == element::i32,
                          "QSAIndexer expects an i32 key_length input, got ",
                          key_length_et);
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::KEY_LENGTH).compatible(ov::PartialShape{batch}),
                          "QSAIndexer expects key_length shaped [B], got ",
                          get_input_partial_shape(InputIdx::KEY_LENGTH));

    if (h_iq.is_static() && h_ik.is_static()) {
        const auto h_iq_value = static_cast<size_t>(h_iq.get_length());
        const auto h_ik_value = static_cast<size_t>(h_ik.get_length());
        NODE_VALIDATION_CHECK(this,
                              h_ik_value != 0 && h_iq_value % h_ik_value == 0,
                              "QSAIndexer: H_iq must be a positive multiple of H_ik.");
        NODE_VALIDATION_CHECK(this,
                              m_config.num_selection_heads == 1 || m_config.num_selection_heads == h_ik_value,
                              "QSAIndexer: num_selection_heads must be 1 or H_ik.");
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_size(OutputIdx::COUNT);
    set_output_type(OutputIdx::SELECTED_BLOCK_INDICES, element::i32, output_shapes[OutputIdx::SELECTED_BLOCK_INDICES]);
    set_output_type(OutputIdx::SELECTED_BLOCK_COUNT, element::i32, output_shapes[OutputIdx::SELECTED_BLOCK_COUNT]);
    for (auto [out, in] : {std::pair{OutputIdx::BLOCK_INDEX_KEYS_CACHE, InputIdx::BLOCK_INDEX_KEYS_CACHE},
                           std::pair{OutputIdx::PAST_INCOMPLETE_RAW_K, InputIdx::PAST_INCOMPLETE_RAW_K},
                           std::pair{OutputIdx::PAST_INCOMPLETE_ROPE, InputIdx::PAST_INCOMPLETE_ROPE}}) {
        set_output_type(out, get_input_element_type(in), output_shapes[out]);
    }
}

std::shared_ptr<Node> QSAIndexer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(internal_QSAIndexer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<QSAIndexer>(new_args, m_config);
}

}  // namespace ov::op::internal
