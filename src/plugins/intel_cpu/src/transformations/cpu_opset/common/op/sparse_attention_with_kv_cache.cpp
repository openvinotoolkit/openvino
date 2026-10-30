// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_attention_with_kv_cache.hpp"

#include <cstddef>
#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"
#include "transformations/itt.hpp"

namespace ov::intel_cpu {

SparseAttentionWithKVCache::SparseAttentionWithKVCache(const OutputVector& args, const Config& cfg)
    : Op(args),
      m_config(cfg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> SparseAttentionWithKVCache::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(SparseAttentionWithKVCache_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SparseAttentionWithKVCache>(new_args, m_config);
}

void SparseAttentionWithKVCache::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(SparseAttentionWithKVCache_validate_and_infer_types);
    const auto input_num = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          input_num == InputIdx::COUNT_MIN || input_num == InputIdx::COUNT_MAX,
                          "The number of inputs must be ",
                          InputIdx::COUNT_MIN,
                          " or ",
                          InputIdx::COUNT_MAX,
                          ". Got: ",
                          input_num);
    NODE_VALIDATION_CHECK(this, m_config.key_block_size > 0, "key_block_size must be positive.");

    const auto beam_idx_idx = input_num - 3;
    const auto past_k_idx = input_num - 2;
    const auto past_v_idx = input_num - 1;

    const auto& q_ps = get_input_partial_shape(InputIdx::QUERY);
    const auto& cur_k_ps = get_input_partial_shape(InputIdx::CUR_KEY);
    const auto& cur_v_ps = get_input_partial_shape(InputIdx::CUR_VALUE);
    auto past_k_ps = get_input_partial_shape(past_k_idx);
    auto past_v_ps = get_input_partial_shape(past_v_idx);
    const auto& beam_idx_ps = get_input_partial_shape(beam_idx_idx);

    NODE_VALIDATION_CHECK(this, q_ps.rank().is_dynamic() || q_ps.size() == 4, "query must be 4D [N, H, L, E].");
    NODE_VALIDATION_CHECK(this, cur_k_ps.rank().is_dynamic() || cur_k_ps.size() == 4, "key must be 4D.");
    NODE_VALIDATION_CHECK(this, cur_v_ps.rank().is_dynamic() || cur_v_ps.size() == 4, "value must be 4D.");
    NODE_VALIDATION_CHECK(this, past_k_ps.rank().is_dynamic() || past_k_ps.size() == 4, "past key must be 4D.");
    NODE_VALIDATION_CHECK(this, past_v_ps.rank().is_dynamic() || past_v_ps.size() == 4, "past value must be 4D.");

    // present = concat(gather(past, beam_idx, axis 0), cur, axis 2): the batch comes from beam_idx.
    auto make_present = [&](ov::PartialShape past_ps, const ov::PartialShape& cur_ps) {
        if (past_ps.rank().is_dynamic()) {
            return past_ps;
        }
        if (beam_idx_ps.rank().is_static() && beam_idx_ps.size() == 1) {
            past_ps[0] = beam_idx_ps[0];
        }
        if (cur_ps.rank().is_static()) {
            past_ps[2] += cur_ps[2];
        } else {
            past_ps[2] = ov::Dimension::dynamic();
        }
        return past_ps;
    };
    auto present_k_ps = make_present(past_k_ps, cur_k_ps);
    auto present_v_ps = make_present(past_v_ps, cur_v_ps);

    ov::PartialShape out_ps = q_ps;
    if (out_ps.rank().is_static() && cur_v_ps.rank().is_static()) {
        out_ps[3] = cur_v_ps[3];
    }
    set_output_type(0, get_input_element_type(InputIdx::QUERY), out_ps);
    set_output_type(1, get_input_element_type(past_k_idx), present_k_ps);
    set_output_type(2, get_input_element_type(past_v_idx), present_v_ps);
}

bool SparseAttentionWithKVCache::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(SparseAttentionWithKVCache_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("key_block_size", m_config.key_block_size);
    visitor.on_attribute("query_block_size", m_config.query_block_size);
    visitor.on_attribute("causal", m_config.causal);
    visitor.finish_structure();
    return true;
}

}  // namespace ov::intel_cpu
