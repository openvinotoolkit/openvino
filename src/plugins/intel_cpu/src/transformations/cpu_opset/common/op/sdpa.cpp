// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa.hpp"

#include <algorithm>
#include <utility>

#include "transformations/itt.hpp"

ov::intel_cpu::ScaledDotProductAttentionWithKVCache::ScaledDotProductAttentionWithKVCache(const OutputVector& args,
                                                                                          Config cfg)
    : Op(args),
      m_config(std::move(cfg)) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::ScaledDotProductAttentionWithKVCache::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(ScaledDotProductAttentionWithKVCache_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(new_args, m_config);
}

void ov::intel_cpu::ScaledDotProductAttentionWithKVCache::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(ScaledDotProductAttentionWithKVCache_validate_and_infer_types);
    auto input_num = get_input_size();
    // [B, H, L1, S]
    auto q_ps = get_input_partial_shape(0);
    // [B, H, L0, S]
    auto past_k_ps = get_input_partial_shape(input_num - 2);
    auto past_v_ps = get_input_partial_shape(input_num - 1);
    // [present_kv_batch_size]
    auto beam_idx_ps = get_input_partial_shape(input_num - 3);

    auto output_logits = q_ps;
    NODE_VALIDATION_CHECK(this, m_config.output_BLHxS == false);
    NODE_VALIDATION_CHECK(this, q_ps.rank().is_static());
    NODE_VALIDATION_CHECK(this, q_ps.size() >= 3);
    // permute_axes from original to [B, H, L, S]
    const auto& permute_axes = this->m_config.permute_axes;
    if (past_k_ps.rank().is_static() || past_v_ps.rank().is_static()) {
        const size_t batch_index = permute_axes.empty() ? 0 : permute_axes[0];
        const size_t length_index = permute_axes.empty() ? q_ps.size() - 2 : permute_axes[permute_axes.size() - 2];
        const size_t head_num_index = permute_axes.empty() ? q_ps.size() - 3 : permute_axes[permute_axes.size() - 3];
        if (past_k_ps.rank().is_static()) {
            NODE_VALIDATION_CHECK(this, q_ps.size() == past_k_ps.size());
        }
        if (past_v_ps.rank().is_static()) {
            NODE_VALIDATION_CHECK(this, q_ps.size() == past_v_ps.size());
        }
        for (size_t i = 0; i < q_ps.size(); i++) {
            if (i == head_num_index) {
                if (q_ps[i].is_static() && past_v_ps[i].is_static()) {
                    NODE_VALIDATION_CHECK(this,
                                          q_ps[i].get_length() % past_v_ps[i].get_length() == 0,
                                          "shape not compatiable at index ",
                                          i);
                }
                if (past_k_ps[i].is_static() && past_v_ps[i].is_static()) {
                    NODE_VALIDATION_CHECK(this,
                                          past_k_ps[i].get_length() == past_v_ps[i].get_length(),
                                          "kv shape not compatiable at index ",
                                          i);
                }
            } else {
                continue;
            }
        }
        // batch_size can be dynamically changed by gather logic
        if (past_k_ps.rank().is_static()) {
            past_k_ps[batch_index] = beam_idx_ps[0];
            past_k_ps[length_index] += q_ps[length_index];
        }
        if (past_v_ps.rank().is_static()) {
            past_v_ps[batch_index] = beam_idx_ps[0];
            past_v_ps[length_index] += q_ps[length_index];
        }
    }
    if (!permute_axes.empty()) {
        if (q_ps.rank().is_static()) {
            // q_ps needs permute to BHLS
            for (size_t i = 0; i < q_ps.size(); i++) {
                output_logits[i] = q_ps[permute_axes[i]];
            }
        }
    }
    if (output_logits.rank().is_static() && past_v_ps.rank().is_static()) {
        output_logits[output_logits.size() - 1] = past_v_ps[output_logits.size() - 1];
    }
    set_output_type(0, get_input_element_type(0), output_logits);
    set_output_type(1, get_input_element_type(input_num - 1), past_k_ps);
    set_output_type(2, get_input_element_type(input_num - 1), past_v_ps);
}

bool ov::intel_cpu::ScaledDotProductAttentionWithKVCache::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(ScaledDotProductAttentionWithKVCache_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("output_BLHxS", m_config.output_BLHxS);
    visitor.on_attribute("fuse_causal_attn", m_config.fuse_causal_attn);
    visitor.on_attribute("is_causal", m_config.is_causal);
    visitor.on_attribute("fuse_concat", m_config.fuse_concat);
    visitor.on_attribute("permute_axes", m_config.permute_axes);
    visitor.finish_structure();
    return true;
}

ov::intel_cpu::SDPAWithTransposeReshape::SDPAWithTransposeReshape(const OutputVector& args, Config cfg)
    : Op(args),
      m_config(std::move(cfg)) {}

std::shared_ptr<ov::Node> ov::intel_cpu::SDPAWithTransposeReshape::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(SDPAWithTransposeReshape_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::SDPAWithTransposeReshape>(new_args, m_config);
}

void ov::intel_cpu::SDPAWithTransposeReshape::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(SDPAWithTransposeReshape_validate_and_infer_types);
    // [B,L,H*S]
    auto q_ps = get_input_partial_shape(0);
    const auto& output_ps = q_ps;
    NODE_VALIDATION_CHECK(this, m_config.output_BLHxS == true);
    NODE_VALIDATION_CHECK(this, m_config.input_BLHxS == true);
    NODE_VALIDATION_CHECK(this, q_ps.size() == 3u);

    // permute_axes should be [B, H, L, S]
    const auto& permute_axes = this->m_config.permute_axes;
    NODE_VALIDATION_CHECK(this, permute_axes.size() == 4u);

    // order_HS should be [H,S]
    const auto& order_HS = this->m_config.order_HS;
    NODE_VALIDATION_CHECK(this, order_HS.size() == 2u);

    set_output_type(0, get_input_element_type(0), output_ps);
}

bool ov::intel_cpu::SDPAWithTransposeReshape::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(SDPAWithTransposeReshape_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("input_BLHxS", m_config.input_BLHxS);
    visitor.on_attribute("output_BLHxS", m_config.output_BLHxS);
    visitor.on_attribute("permute_axes", m_config.permute_axes);
    visitor.on_attribute("order_HS", m_config.order_HS);
    visitor.finish_structure();
    return true;
}
