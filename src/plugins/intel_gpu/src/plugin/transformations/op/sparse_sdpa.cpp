// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sparse_sdpa.hpp"

#include <string>
#include <utility>

#include "qsa_op_utils.hpp"

namespace ov::intel_gpu::op {

using namespace qsa_util;

SparseSDPA::SparseSDPA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type)
    : Op(args),
      m_config(config),
      m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

bool SparseSDPA::validate_config(const Config& c, std::string& reason) {
    return validate_sparse_attention_config(c, reason);
}

bool SparseSDPA::visit_attributes(ov::AttributeVisitor& visitor) {
    visit_sparse_attention_config(visitor, m_config);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void SparseSDPA::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          get_input_size() >= InputIdx::COUNT_MIN && get_input_size() <= InputIdx::COUNT_MAX,
                          "SparseSDPA expects ",
                          InputIdx::COUNT_MIN,
                          "..",
                          InputIdx::COUNT_MAX,
                          " inputs, got ",
                          get_input_size());

    std::string reason;
    NODE_VALIDATION_CHECK(this, validate_config(m_config, reason), "SparseSDPA configuration is invalid: ", reason);

    const auto& query_ps = get_input_partial_shape(InputIdx::QUERY);
    NODE_VALIDATION_CHECK(this, query_ps.rank().is_dynamic() || query_ps.size() == 4, "SparseSDPA expects a rank-4 [B, Hq, L, Dh] query input, got ", query_ps);

    // The kernels read the selection plan and key_length as int32_t.
    const std::pair<size_t, int64_t> int_inputs[] = {{InputIdx::SELECTED_BLOCK_INDICES, 4}, {InputIdx::SELECTED_BLOCK_COUNT, 3}, {InputIdx::KEY_LENGTH, 1}};
    for (const auto& [idx, rank] : int_inputs) {
        const auto& et = get_input_element_type(idx);
        const auto& ps = get_input_partial_shape(idx);
        NODE_VALIDATION_CHECK(this, et.is_dynamic() || et == element::i32, "SparseSDPA expects an i32 input ", idx, ", got ", et);
        NODE_VALIDATION_CHECK(this, ps.rank().compatible(rank), "SparseSDPA expects a rank-", rank, " input ", idx, ", got ", ps);
    }
    const auto batch = query_ps.rank().is_static() ? query_ps[0] : ov::Dimension::dynamic();
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::KEY_LENGTH).compatible(ov::PartialShape{batch}),
                          "SparseSDPA expects a [B] key_length input, got ",
                          get_input_partial_shape(InputIdx::KEY_LENGTH));

    // Q/K/V geometry must agree with the config the kernels are specialized for.
    const auto hkv = static_cast<int64_t>(m_config.num_kv_heads);
    const auto sel = static_cast<int64_t>(m_config.num_selection_heads);
    check_shape(this,
                InputIdx::QUERY,
                "query",
                ov::PartialShape{batch, static_cast<int64_t>(m_config.num_heads), dim_at(query_ps, 2), static_cast<int64_t>(m_config.k_head_size)});
    const auto seq = dim_at(get_input_partial_shape(InputIdx::KEY), 2);
    check_shape(this, InputIdx::KEY, "key", ov::PartialShape{batch, hkv, seq, static_cast<int64_t>(m_config.k_head_size)});
    check_shape(this, InputIdx::VALUE, "value", ov::PartialShape{batch, hkv, seq, static_cast<int64_t>(m_config.v_head_size)});
    const auto q_len = dim_at(query_ps, 2);
    const auto k_max = dim_at(get_input_partial_shape(InputIdx::SELECTED_BLOCK_INDICES), 3);
    check_shape(this, InputIdx::SELECTED_BLOCK_INDICES, "selected_block_indices", ov::PartialShape{batch, sel, q_len, k_max});
    check_shape(this, InputIdx::SELECTED_BLOCK_COUNT, "selected_block_count", ov::PartialShape{batch, sel, q_len});
    check_float_inputs(this, {InputIdx::QUERY, InputIdx::KEY, InputIdx::VALUE});
    if (has_scale_input()) {
        const auto& scale_ps = get_input_partial_shape(InputIdx::SCALE);
        NODE_VALIDATION_CHECK(this,
                              scale_ps.is_dynamic() || ov::shape_size(scale_ps.to_shape()) == 1,
                              "SparseSDPA expects a scalar or 1-element scale input, got ",
                              scale_ps);
    }
    if (has_sinks()) {
        NODE_VALIDATION_CHECK(this,
                              get_input_partial_shape(InputIdx::SINKS).compatible(ov::PartialShape{static_cast<int64_t>(m_config.num_heads)}),
                              "SparseSDPA expects a [num_heads] sinks input, got ",
                              get_input_partial_shape(InputIdx::SINKS));
    }

    auto out_ps = ov::PartialShape::dynamic(4);
    if (query_ps.rank().is_static()) {
        out_ps[0] = query_ps[0];
        out_ps[2] = query_ps[2];
    }
    out_ps[1] = ov::Dimension(static_cast<int64_t>(m_config.num_heads));
    out_ps[3] = ov::Dimension(static_cast<int64_t>(m_config.v_head_size));

    const auto out_et = m_output_type == ov::element::dynamic ? get_input_element_type(InputIdx::QUERY) : m_output_type;
    set_output_size(1);
    set_output_type(0, out_et, out_ps);
}

std::shared_ptr<Node> SparseSDPA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SparseSDPA>(new_args, m_config, m_output_type);
}

}  // namespace ov::intel_gpu::op
