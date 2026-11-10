// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/paged_qsa_indexer.hpp"

#include <cmath>
#include <limits>

namespace ov::op::internal {

PagedQSAIndexer::PagedQSAIndexer(const OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

bool PagedQSAIndexer::visit_attributes(AttributeVisitor& visitor) {
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

void PagedQSAIndexer::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == InputIdx::COUNT, "PagedQSAIndexer expects 13 inputs");
    const auto max_dimension = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    NODE_VALIDATION_CHECK(this,
                          m_config.num_selection_heads > 0 && m_config.num_selection_heads <= max_dimension &&
                              m_config.block_topk > 0 && m_config.block_topk <= max_dimension &&
                              m_config.indexer_head_dim > 0,
                          "PagedQSAIndexer requires positive selection head, top-k and head dimensions");
    if (m_config.scale == 0.0f) {
        m_config.scale = 1.0f / std::sqrt(static_cast<float>(m_config.indexer_head_dim));
    }

    auto tokens = Dimension::dynamic();
    for (const auto idx : {InputIdx::INDEX_QUERY, InputIdx::RAW_INDEX_KEYS, InputIdx::ROPE_COS, InputIdx::ROPE_SIN}) {
        const auto& shape = get_input_partial_shape(idx);
        NODE_VALIDATION_CHECK(this, shape.rank().compatible(2), "PagedQSAIndexer expects rank-2 activations");
        if (shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  Dimension::merge(tokens, tokens, shape[0]),
                                  "PagedQSAIndexer activation token counts must match");
        }
    }
    const Dimension heads(static_cast<int64_t>(m_config.num_selection_heads));
    const Dimension topk(static_cast<int64_t>(m_config.block_topk));
    set_output_size(OutputIdx::COUNT);
    set_output_type(OutputIdx::SELECTED_BLOCK_INDICES, element::i32, PartialShape{tokens, heads, topk});
    set_output_type(OutputIdx::SELECTED_BLOCK_COUNT, element::i32, PartialShape{tokens, heads});
}

std::shared_ptr<Node> PagedQSAIndexer::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<PagedQSAIndexer>(new_args, m_config);
}

}  // namespace ov::op::internal
