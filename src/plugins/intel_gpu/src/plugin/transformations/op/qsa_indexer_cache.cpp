// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/qsa_indexer_cache.hpp"

#include "openvino/core/partial_shape.hpp"

namespace ov::intel_gpu::op {

QSAIndexerCache::QSAIndexerCache(const std::shared_ptr<ov::op::util::Variable>& raw_k_variable,
                                 const std::shared_ptr<ov::op::util::Variable>& summary_variable,
                                 const Config& config)
    : Op(ov::OutputVector{})
    , m_summary_variable(summary_variable)
    , m_config(config) {
    m_variable = raw_k_variable;
    validate_and_infer_types();
}

bool QSAIndexerCache::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("heads_num", m_config.heads_num);
    visitor.on_attribute("kv_heads_num", m_config.kv_heads_num);
    visitor.on_attribute("k_head_size", m_config.k_head_size);
    visitor.on_attribute("v_head_size", m_config.v_head_size);
    visitor.on_attribute("hidden_size", m_config.hidden_size);
    visitor.on_attribute("indexer_heads", m_config.indexer_heads);
    visitor.on_attribute("indexer_kv_heads", m_config.indexer_kv_heads);
    visitor.on_attribute("indexer_head_dim", m_config.indexer_head_dim);
    visitor.on_attribute("indexer_rotary_dim", m_config.indexer_rotary_dim);
    visitor.on_attribute("budget", m_config.budget);
    visitor.on_attribute("compress_ratio", m_config.compress_ratio);
    visitor.on_attribute("qsa_block_tokens", m_config.qsa_block_tokens);
    visitor.on_attribute("pa_block_size", m_config.pa_block_size);
    visitor.on_attribute("block_topk", m_config.block_topk);
    visitor.on_attribute("scale", m_config.scale);
    visitor.on_attribute("indexer_norm_eps", m_config.indexer_norm_eps);
    visitor.on_attribute("has_output_gate", m_config.has_output_gate);
    visitor.on_attribute("is_causal", m_config.is_causal);
    return true;
}

void QSAIndexerCache::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 0, "QSAIndexerCache is a pure state declaration and expects 0 inputs, got ", get_input_size());
    OPENVINO_ASSERT(m_variable, "QSAIndexerCache requires a raw-K Variable");
    OPENVINO_ASSERT(m_summary_variable, "QSAIndexerCache requires a summary Variable");

    const auto& raw_k_info = m_variable->get_info();
    const auto& summary_info = m_summary_variable->get_info();

    set_output_size(2);
    set_output_type(0, raw_k_info.data_type, raw_k_info.data_shape);
    set_output_type(1, summary_info.data_type, summary_info.data_shape);
}

std::shared_ptr<Node> QSAIndexerCache::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<QSAIndexerCache>(m_variable, m_summary_variable, m_config);
}

}  // namespace ov::intel_gpu::op
