// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/qsa_indexer_in_place.hpp"

#include <string>
#include <utility>

namespace ov::intel_gpu::op {

QSAIndexerInPlace::QSAIndexerInPlace(const ov::OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

bool QSAIndexerInPlace::visit_attributes(ov::AttributeVisitor& visitor) {
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

void QSAIndexerInPlace::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == InputIdx::COUNT, "QSAIndexerInPlace expects ", InputIdx::COUNT, " inputs, got ", get_input_size());
    std::string reason;
    NODE_VALIDATION_CHECK(this, ov::op::internal::QSAIndexer::validate_config(m_config, reason), "QSAIndexerInPlace configuration is invalid: ", reason);

    const auto& index_query_ps = get_input_partial_shape(InputIdx::INDEX_QUERY);
    NODE_VALIDATION_CHECK(this,
                          index_query_ps.rank().is_dynamic() || index_query_ps.size() == 4,
                          "QSAIndexerInPlace expects a rank-4 [B, L, H_iq, D_idx] index_query input, got ",
                          index_query_ps);
    const ov::Dimension head_dim(static_cast<int64_t>(m_config.indexer_head_dim));
    const auto batch = index_query_ps.rank().is_static() ? index_query_ps[0] : ov::Dimension::dynamic();
    const auto length = index_query_ps.rank().is_static() ? index_query_ps[1] : ov::Dimension::dynamic();
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::BLOCK_INDEX_KEYS_CACHE).compatible(ov::PartialShape{batch, -1, head_dim}),
                          "QSAIndexerInPlace expects block_index_keys_cache shaped [B, blocks, indexer_head_dim], got ",
                          get_input_partial_shape(InputIdx::BLOCK_INDEX_KEYS_CACHE));

    const ov::Dimension selection_heads(static_cast<int64_t>(m_config.num_selection_heads));
    set_output_size(OutputIdx::COUNT);
    set_output_type(OutputIdx::SELECTED_BLOCK_INDICES,
                    ov::element::i32,
                    ov::PartialShape{batch, selection_heads, length, ov::Dimension(static_cast<int64_t>(m_config.block_topk))});
    set_output_type(OutputIdx::SELECTED_BLOCK_COUNT, ov::element::i32, ov::PartialShape{batch, selection_heads, length});
    set_output_type(OutputIdx::BLOCK_INDEX_KEYS_CACHE, get_input_element_type(InputIdx::BLOCK_INDEX_KEYS_CACHE), ov::PartialShape{0});
    for (auto [out, in] : {std::pair{OutputIdx::PAST_INCOMPLETE_RAW_K, InputIdx::PAST_INCOMPLETE_RAW_K},
                           std::pair{OutputIdx::PAST_INCOMPLETE_ROPE, InputIdx::PAST_INCOMPLETE_ROPE}}) {
        set_output_type(out, get_input_element_type(in), get_input_partial_shape(in));
    }
}

std::shared_ptr<ov::Node> QSAIndexerInPlace::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<QSAIndexerInPlace>(new_args, m_config);
}

}  // namespace ov::intel_gpu::op
