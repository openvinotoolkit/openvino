// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa.hpp"

#include <algorithm>

#include "transformations/itt.hpp"

ov::intel_cpu::ScaledDotProductAttentionWithKVCache::ScaledDotProductAttentionWithKVCache(const OutputVector& args, const Config& cfg)
    : Op(args),
      m_config(cfg) {
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
    auto past_kv_ps = get_input_partial_shape(input_num - 1);

    auto output_logits = q_ps;
    NODE_VALIDATION_CHECK(this, m_config.output_BLHxS == false);
    NODE_VALIDATION_CHECK(this, q_ps.size() >= 3);
    const size_t length_index = this->m_config.is_lbhs_input ? 0 : q_ps.size() - 2;
    const size_t head_num_index = this->m_config.is_lbhs_input ? 2 : q_ps.size() - 3;
    if (past_kv_ps.rank().is_static()) {
        NODE_VALIDATION_CHECK(this, q_ps.size() == past_kv_ps.size());
        for (size_t i = 0; i < q_ps.size(); i++) {
            if (i == head_num_index) {
                if (q_ps[i].is_static() && past_kv_ps[i].is_static()) {
                    NODE_VALIDATION_CHECK(this,
                                          q_ps[i].get_length() % past_kv_ps[i].get_length() == 0,
                                          "shape not compatiable at index ",
                                          i);
                } else if (i == length_index) {
                    continue;
                } else {
                    NODE_VALIDATION_CHECK(this,
                                          q_ps[i].compatible(past_kv_ps[i]),
                                          "shape not compatiable at index ",
                                          i);
                }
            }
        }
        if (this->m_config.is_lbhs_input) {
                past_kv_ps[0] += q_ps[0];
        } else {
                past_kv_ps[q_ps.size() - 2] += q_ps[q_ps.size() - 2];
        }
    }
    if (this->m_config.is_lbhs_input) {
        if (q_ps.rank().is_static()) {
            if (q_ps.rank().get_length() == 4) {
                output_logits = PartialShape{q_ps[1], q_ps[2], q_ps[0], q_ps[3]};
            } else {
                NODE_VALIDATION_CHECK(this, q_ps.size() == 4, "LBHS input must have rank 4");
            }
        }
    }
    set_output_type(0, get_input_element_type(0), output_logits);
    set_output_type(1, get_input_element_type(input_num - 1), past_kv_ps);
    set_output_type(2, get_input_element_type(input_num - 1), past_kv_ps);
}

bool ov::intel_cpu::ScaledDotProductAttentionWithKVCache::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(ScaledDotProductAttentionWithKVCache_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("output_BLHxS", m_config.output_BLHxS);
    visitor.on_attribute("fuse_causal_attn", m_config.fuse_causal_attn);
    visitor.on_attribute("is_causal", m_config.is_causal);
    visitor.on_attribute("fuse_concat", m_config.fuse_concat);
    visitor.on_attribute("is_lbhs_input", m_config.is_lbhs_input);
    visitor.on_attribute("is_multi_query", m_config.is_multi_query);
    visitor.finish_structure();
    return true;
}
