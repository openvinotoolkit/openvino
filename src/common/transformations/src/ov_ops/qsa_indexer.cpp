// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/qsa_indexer.hpp"

#include <limits>
#include <utility>

namespace ov::op::internal {

QSAIndexer::QSAIndexer(const OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

bool QSAIndexer::visit_attributes(AttributeVisitor& visitor) {
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
    NODE_VALIDATION_CHECK(this, get_input_size() == InputIdx::COUNT, "QSAIndexer expects 10 inputs");
    const auto& query_shape = get_input_partial_shape(InputIdx::INDEX_QUERY);
    NODE_VALIDATION_CHECK(this, query_shape.rank().compatible(4), "QSAIndexer expects a rank-4 query");
    const auto max_dimension = static_cast<size_t>(std::numeric_limits<int64_t>::max());
    NODE_VALIDATION_CHECK(this,
                          m_config.num_selection_heads > 0 && m_config.num_selection_heads <= max_dimension &&
                              m_config.block_topk > 0 && m_config.block_topk <= max_dimension,
                          "QSAIndexer requires positive selection head and top-k dimensions");

    const auto batch = query_shape.rank().is_static() ? query_shape[0] : Dimension::dynamic();
    const auto length = query_shape.rank().is_static() ? query_shape[1] : Dimension::dynamic();
    const Dimension heads(static_cast<int64_t>(m_config.num_selection_heads));
    const Dimension topk(static_cast<int64_t>(m_config.block_topk));
    set_output_size(OutputIdx::COUNT);
    set_output_type(OutputIdx::SELECTED_BLOCK_INDICES, element::i32, PartialShape{batch, heads, length, topk});
    set_output_type(OutputIdx::SELECTED_BLOCK_COUNT, element::i32, PartialShape{batch, heads, length});
    for (const auto& [out, in] : {std::pair{OutputIdx::BLOCK_INDEX_KEYS_CACHE, InputIdx::BLOCK_INDEX_KEYS_CACHE},
                                  std::pair{OutputIdx::PAST_INCOMPLETE_RAW_K, InputIdx::PAST_INCOMPLETE_RAW_K},
                                  std::pair{OutputIdx::PAST_INCOMPLETE_ROPE, InputIdx::PAST_INCOMPLETE_ROPE}}) {
        set_output_type(out, get_input_element_type(in), get_input_partial_shape(in));
    }
}

std::shared_ptr<Node> QSAIndexer::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<QSAIndexer>(new_args, m_config);
}

}  // namespace ov::op::internal
