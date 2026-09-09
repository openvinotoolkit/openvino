// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/qsa.hpp"

#include <string>

namespace ov::intel_gpu::op {

QSA::QSA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type)
    : Op(args),
      m_config(config),
      m_output_type(output_type) {
    validate_and_infer_types();
}

bool QSA::validate_config(const Config& c, std::string& reason) {
    auto fail = [&reason](const char* msg) {
        reason = msg;
        return false;
    };

    if (c.compress_ratio == 0) {
        return fail("compress_ratio == 0 means the layer is not a QSA layer");
    }
    if (c.indexer_kv_heads != 1) {
        return fail("indexer_kv_heads must be 1 (indexer is MQA)");
    }
    if (c.indexer_heads == 0 || c.indexer_head_dim == 0) {
        return fail("indexer_heads / indexer_head_dim must be positive");
    }
    if (c.indexer_rotary_dim > c.indexer_head_dim) {
        return fail("indexer_rotary_dim must not exceed indexer_head_dim");
    }
    if (c.indexer_rotary_dim % 2 != 0) {
        return fail("indexer_rotary_dim must be even (rotate_half pairs channel j with j + rotary_dim/2)");
    }
    if (c.budget == 0 || c.budget % c.compress_ratio != 0) {
        return fail("budget must be positive and divisible by compress_ratio");
    }
    if (c.block_topk != c.budget / c.compress_ratio) {
        return fail("block_topk must equal budget / compress_ratio");
    }
    if (c.qsa_block_tokens != c.compress_ratio) {
        return fail("qsa_block_tokens must equal compress_ratio");
    }
    if (c.pa_block_size == 0 || c.pa_block_size % c.compress_ratio != 0) {
        // Required so that one PA page holds an integral number of complete QSA blocks and
        // the summary cache can share the main page table (see design 7.1.2).
        return fail("pa_block_size must be positive and divisible by compress_ratio");
    }
    if (c.heads_num == 0 || c.kv_heads_num == 0 || c.heads_num % c.kv_heads_num != 0) {
        return fail("heads_num must be a positive multiple of kv_heads_num");
    }
    if (c.k_head_size == 0 || c.k_head_size % 16 != 0 || c.v_head_size == 0 || c.v_head_size % 16 != 0) {
        return fail("k_head_size / v_head_size must be positive multiples of 16");
    }
    if (c.indexer_head_dim % 16 != 0) {
        return fail("indexer_head_dim must be a multiple of 16");
    }
    if (c.hidden_size == 0) {
        return fail("hidden_size must be positive");
    }
    return true;
}

bool QSA::visit_attributes(ov::AttributeVisitor& visitor) {
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

void QSA::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == InputIdx::COUNT,
                          "QSA expects ",
                          InputIdx::COUNT,
                          " inputs, got ",
                          get_input_size());

    std::string reason;
    NODE_VALIDATION_CHECK(this, validate_config(m_config, reason), "QSA configuration is invalid: ", reason);

    const auto& query_ps = get_input_partial_shape(InputIdx::QUERY);
    NODE_VALIDATION_CHECK(this,
                          query_ps.rank().is_dynamic() || query_ps.rank().get_length() == 2,
                          "QSA expects a rank-2 [tokens, Hq*Dh] query input.");

    auto out_ps = ov::PartialShape::dynamic(2);
    if (query_ps.rank().is_static()) {
        out_ps[0] = query_ps[0];
    }
    out_ps[1] = ov::Dimension(static_cast<int64_t>(m_config.heads_num * m_config.v_head_size));

    const auto out_et = m_output_type == ov::element::dynamic ? get_input_element_type(InputIdx::QUERY) : m_output_type;
    set_output_size(1);
    set_output_type(0, out_et, out_ps);
}

std::shared_ptr<Node> QSA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<QSA>(new_args, m_config, m_output_type);
}

}  // namespace ov::intel_gpu::op
