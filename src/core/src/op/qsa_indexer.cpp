// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"

#include <cmath>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov::op::internal {

QSAIndexer::QSAIndexer(const ov::OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

bool QSAIndexer::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(QSAIndexer_visit_attributes);
    visitor.on_attribute("key_block_size", m_config.key_block_size);
    visitor.on_attribute("token_budget", m_config.token_budget);
    visitor.on_attribute("num_selection_heads", m_config.num_selection_heads);
    visitor.on_attribute("q_epsilon", m_config.q_epsilon);
    visitor.on_attribute("k_epsilon", m_config.k_epsilon);
    visitor.on_attribute("is_causal", m_config.is_causal);
    return true;
}

void QSAIndexer::validate_and_infer_types() {
    OV_OP_SCOPE(QSAIndexer_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 10, "QSAIndexer requires ten inputs.");
    NODE_VALIDATION_CHECK(this, m_config.key_block_size > 0);
    NODE_VALIDATION_CHECK(this, m_config.token_budget > 0 && m_config.token_budget % m_config.key_block_size == 0);
    NODE_VALIDATION_CHECK(this, m_config.num_selection_heads > 0);
    NODE_VALIDATION_CHECK(this,
                          std::isfinite(m_config.q_epsilon) && m_config.q_epsilon > 0.0f &&
                              std::isfinite(m_config.k_epsilon) && m_config.k_epsilon > 0.0f);
    NODE_VALIDATION_CHECK(this, m_config.is_causal, "Only causal block selection is supported.");

    auto data_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_type.is_dynamic() || data_type.is_real(),
                          "QSAIndexer data inputs must have a floating-point element type.");
    for (const auto i : {1U, 2U, 3U, 5U, 6U, 7U, 8U, 9U}) {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(data_type, data_type, get_input_element_type(i)),
                              "QSAIndexer floating-point input element types must match.");
    }
    NODE_VALIDATION_CHECK(this,
                          data_type.is_dynamic() || data_type.is_real(),
                          "QSAIndexer data inputs must have a floating-point element type.");
    const auto length_type = get_input_element_type(4);
    NODE_VALIDATION_CHECK(this,
                          length_type.is_dynamic() || length_type == element::i32 || length_type == element::i64,
                          "key_length must have i32 or i64 element type.");

    const auto query_shape = get_input_partial_shape(0);
    const auto raw_keys_shape = get_input_partial_shape(1);
    const auto rope_cos_shape = get_input_partial_shape(2);
    const auto rope_sin_shape = get_input_partial_shape(3);
    const auto length_shape = get_input_partial_shape(4);
    const auto q_weight_shape = get_input_partial_shape(5);
    const auto k_weight_shape = get_input_partial_shape(6);
    const auto cache_shape = get_input_partial_shape(7);
    const auto incomplete_keys_shape = get_input_partial_shape(8);
    const auto incomplete_rope_shape = get_input_partial_shape(9);
    const auto rope_disabled = ov::util::is_empty_constant_tensor(input_value(2));
    NODE_VALIDATION_CHECK(this,
                          ov::util::is_empty_constant_tensor(input_value(3)) == rope_disabled &&
                              ov::util::is_empty_constant_tensor(input_value(9)) == rope_disabled,
                          "rope_cos, rope_sin, and past_incomplete_rope must all be empty constants to disable RoPE.");
    const auto q_norm_disabled = ov::util::is_empty_constant_tensor(input_value(5));
    const auto k_norm_disabled = ov::util::is_empty_constant_tensor(input_value(6));
    NODE_VALIDATION_CHECK(this, query_shape.rank().compatible(4));
    NODE_VALIDATION_CHECK(this, raw_keys_shape.rank().compatible(3));
    if (!rope_disabled) {
        NODE_VALIDATION_CHECK(this, rope_cos_shape.rank().compatible(3));
        NODE_VALIDATION_CHECK(this, rope_sin_shape.rank().compatible(3));
    }
    NODE_VALIDATION_CHECK(this, length_shape.rank().compatible(1));
    NODE_VALIDATION_CHECK(this, q_norm_disabled || q_weight_shape.rank().compatible(1));
    NODE_VALIDATION_CHECK(this, k_norm_disabled || k_weight_shape.rank().compatible(1));
    NODE_VALIDATION_CHECK(this, cache_shape.rank().compatible(3));
    NODE_VALIDATION_CHECK(this, incomplete_keys_shape.rank().compatible(3));
    if (!rope_disabled) {
        NODE_VALIDATION_CHECK(this, incomplete_rope_shape.rank().compatible(2));
    }

    auto batch = Dimension::dynamic();
    auto sequence = Dimension::dynamic();
    auto depth = Dimension::dynamic();
    auto heads = Dimension::dynamic();
    if (query_shape.rank().is_static()) {
        batch = query_shape[0];
        sequence = query_shape[2];
        depth = query_shape[3];
        heads = query_shape[1];
        NODE_VALIDATION_CHECK(this, heads.compatible(4));
        NODE_VALIDATION_CHECK(this, depth.compatible(128));
    }
    if (raw_keys_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, raw_keys_shape[0].compatible(batch));
        NODE_VALIDATION_CHECK(this, raw_keys_shape[2].compatible(depth));
        if (query_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(this, sequence.compatible(raw_keys_shape[1]));
        }
    }
    if (cache_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, cache_shape[0].compatible(batch));
        NODE_VALIDATION_CHECK(this, cache_shape[2].compatible(depth));
    }
    if (incomplete_keys_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, incomplete_keys_shape[0].compatible(batch));
        NODE_VALIDATION_CHECK(this, incomplete_keys_shape[1].compatible(m_config.key_block_size - 1));
        NODE_VALIDATION_CHECK(this, incomplete_keys_shape[2].compatible(depth));
    }
    if (length_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, length_shape[0].compatible(1) || length_shape[0].compatible(batch));
    }
    if (!q_norm_disabled && q_weight_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, q_weight_shape[0].compatible(depth));
    }
    if (!k_norm_disabled && k_weight_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, k_weight_shape[0].compatible(depth));
    }
    if (!rope_disabled && rope_cos_shape.rank().is_static() && rope_sin_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, rope_cos_shape[0].compatible(1) || rope_cos_shape[0].compatible(batch));
        NODE_VALIDATION_CHECK(this, rope_sin_shape[0].compatible(1) || rope_sin_shape[0].compatible(batch));
        NODE_VALIDATION_CHECK(
            this,
            rope_cos_shape[1].compatible(raw_keys_shape.rank().is_static() ? raw_keys_shape[1] : Dimension::dynamic()));
        NODE_VALIDATION_CHECK(
            this,
            rope_sin_shape[1].compatible(raw_keys_shape.rank().is_static() ? raw_keys_shape[1] : Dimension::dynamic()));
        NODE_VALIDATION_CHECK(this, rope_cos_shape[2].compatible(rope_sin_shape[2]));
        NODE_VALIDATION_CHECK(this, rope_cos_shape[2].is_dynamic() || rope_cos_shape[2].get_length() <= 128);
        if (incomplete_rope_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(this, incomplete_rope_shape[0].compatible(batch));
            if (rope_cos_shape[2].is_static()) {
                NODE_VALIDATION_CHECK(
                    this,
                    incomplete_rope_shape[1].compatible(Dimension(2 * rope_cos_shape[2].get_length())));
            }
        }
    }

    if (query_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this,
                              heads.is_dynamic() || heads.get_length() % m_config.num_selection_heads == 0,
                              "The query head count must be divisible by num_selection_heads.");
    }
    const auto queries = query_shape.rank().is_static() ? query_shape[2] : Dimension::dynamic();
    const auto selections = Dimension(m_config.num_selection_heads);
    const auto top_k = Dimension(m_config.token_budget / m_config.key_block_size);
    auto updated_rope_shape = incomplete_rope_shape;
    if (rope_disabled) {
        const auto rope_batch = raw_keys_shape.rank().is_static() ? raw_keys_shape[0] : Dimension::dynamic();
        updated_rope_shape = PartialShape{rope_batch, Dimension(0)};
    }
    set_output_type(0, element::i32, PartialShape{batch, selections, queries, top_k});
    set_output_type(1, element::i32, PartialShape{batch, selections, queries});
    set_output_type(2, data_type, cache_shape);
    set_output_type(3, data_type, incomplete_keys_shape);
    set_output_type(4, data_type, updated_rope_shape);
}

std::shared_ptr<Node> QSAIndexer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(QSAIndexer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<QSAIndexer>(new_args, m_config);
}

}  // namespace ov::op::internal
