// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_sdpa.hpp"

#include <string>

#include "itt.hpp"

namespace ov::op::internal {

QSASDPA::QSASDPA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type)
    : Op(args),
      m_config(config),
      m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

bool QSASDPA::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_QSASDPA_visit_attributes);
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
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void QSASDPA::validate_and_infer_types() {
    OV_OP_SCOPE(internal_QSASDPA_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == InputIdx::COUNT, "QSASDPA expects ", InputIdx::COUNT, " inputs, got ", get_input_size());

    std::string reason;
    NODE_VALIDATION_CHECK(this, validate_config(m_config, reason), "QSASDPA configuration is invalid: ", reason);

    const auto& query_ps = get_input_partial_shape(InputIdx::QUERY);
    NODE_VALIDATION_CHECK(this, query_ps.rank().is_dynamic() || query_ps.rank().get_length() == 2, "QSASDPA expects a rank-2 [tokens, Hq*Dh] query input.");

    auto out_ps = ov::PartialShape::dynamic(2);
    if (query_ps.rank().is_static()) {
        out_ps[0] = query_ps[0];
    }
    out_ps[1] = ov::Dimension(static_cast<int64_t>(m_config.heads_num * m_config.v_head_size));

    const auto out_et = m_output_type == ov::element::dynamic ? get_input_element_type(InputIdx::QUERY) : m_output_type;
    set_output_size(1);
    set_output_type(0, out_et, out_ps);
}

std::shared_ptr<Node> QSASDPA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<QSASDPA>(new_args, m_config, m_output_type);
}

}  // namespace ov::op::internal
