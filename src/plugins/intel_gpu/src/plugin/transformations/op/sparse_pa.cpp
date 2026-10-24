// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sparse_pa.hpp"

#include <string>
#include <utility>

#include "qsa_op_utils.hpp"

namespace ov::intel_gpu::op {

using namespace qsa_util;

SparsePA::SparsePA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type)
    : Op(args),
      m_config(config),
      m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

bool SparsePA::validate_config(const Config& c, std::string& reason) {
    return validate_sparse_attention_config(c, reason);
}

bool SparsePA::visit_attributes(ov::AttributeVisitor& visitor) {
    visit_sparse_attention_config(visitor, m_config);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void SparsePA::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          get_input_size() >= InputIdx::COUNT_MIN && get_input_size() <= InputIdx::COUNT_MAX,
                          "SparsePA expects ",
                          InputIdx::COUNT_MIN,
                          "..",
                          InputIdx::COUNT_MAX,
                          " inputs, got ",
                          get_input_size());

    std::string reason;
    NODE_VALIDATION_CHECK(this, validate_config(m_config, reason), "SparsePA configuration is invalid: ", reason);

    const auto& query_ps = get_input_partial_shape(InputIdx::QUERY);
    NODE_VALIDATION_CHECK(this,
                           query_ps.rank().is_dynamic() || query_ps.size() == 2,
                           "SparsePA expects a rank-2 [tokens, num_heads*k_head_size] query input, got ",
                           query_ps);

    // The kernels read every integer input as int32_t.
    const std::pair<size_t, int64_t> int_inputs[] = {{InputIdx::PAST_LENS, 1},
                                                     {InputIdx::SUBSEQUENCE_BEGINS, 1},
                                                     {InputIdx::BLOCK_INDICES, 1},
                                                     {InputIdx::BLOCK_INDICES_BEGINS, 1},
                                                     {InputIdx::SELECTED_BLOCK_INDICES, 3},
                                                     {InputIdx::SELECTED_BLOCK_COUNT, 2}};
    for (const auto& [idx, rank] : int_inputs) {
        const auto& et = get_input_element_type(idx);
        const auto& ps = get_input_partial_shape(idx);
        NODE_VALIDATION_CHECK(this, et.is_dynamic() || et == element::i32, "SparsePA expects an i32 input ", idx, ", got ", et);
        NODE_VALIDATION_CHECK(this, ps.rank().compatible(rank), "SparsePA expects a rank-", rank, " input ", idx, ", got ", ps);
    }

    if (get_input_size() > InputIdx::QB_START) {
        const auto& et = get_input_element_type(InputIdx::QB_START);
        const auto& ps = get_input_partial_shape(InputIdx::QB_START);
        NODE_VALIDATION_CHECK(this, et.is_dynamic() || et == element::i32, "SparsePA expects an i32 qb_start input, got ", et);
        NODE_VALIDATION_CHECK(this, ps.rank().compatible(1), "SparsePA expects a rank-1 qb_start input, got ", ps);
    }

    if (get_input_size() > InputIdx::SELECTED_BLOCK_MASK) {
        const auto& et = get_input_element_type(InputIdx::SELECTED_BLOCK_MASK);
        const auto& ps = get_input_partial_shape(InputIdx::SELECTED_BLOCK_MASK);
        // The GPU plugin stores boolean inputs as u8.
        NODE_VALIDATION_CHECK(this,
                              et.is_dynamic() || et == element::boolean || et == element::u8,
                              "SparsePA expects a boolean selected_block_mask input, got ",
                              et);
        NODE_VALIDATION_CHECK(this, ps.rank().compatible(3), "SparsePA expects a rank-3 selected_block_mask input, got ", ps);
    }

    // Token-major Q/K/V and the page-table vectors must agree with the config and with each other.
    const auto tokens = dim_at(query_ps, 0);
    const auto hq_dk = static_cast<int64_t>(m_config.num_heads * m_config.k_head_size);
    const auto hkv_dk = static_cast<int64_t>(m_config.num_kv_heads * m_config.k_head_size);
    const auto hkv_dv = static_cast<int64_t>(m_config.num_kv_heads * m_config.v_head_size);
    const auto hkv = static_cast<int64_t>(m_config.num_kv_heads);
    check_shape(this, InputIdx::QUERY, "query", ov::PartialShape{tokens, hq_dk});
    check_shape(this, InputIdx::KEY, "key", ov::PartialShape{tokens, hkv_dk});
    check_shape(this, InputIdx::VALUE, "value", ov::PartialShape{tokens, hkv_dv});
    check_shape(this,
                InputIdx::KEY_CACHE,
                "key_cache",
                ov::PartialShape{dim_at(get_input_partial_shape(InputIdx::KEY_CACHE), 0), hkv, dim_at(get_input_partial_shape(InputIdx::KEY_CACHE), 2), static_cast<int64_t>(m_config.k_head_size)});
    check_shape(this,
                InputIdx::VALUE_CACHE,
                "value_cache",
                ov::PartialShape{dim_at(get_input_partial_shape(InputIdx::VALUE_CACHE), 0), hkv, dim_at(get_input_partial_shape(InputIdx::VALUE_CACHE), 2), static_cast<int64_t>(m_config.v_head_size)});
    const auto num_seqs = dim_at(get_input_partial_shape(InputIdx::PAST_LENS), 0);
    const auto seq_begins = num_seqs + 1;
    check_shape(this, InputIdx::SUBSEQUENCE_BEGINS, "subsequence_begins", ov::PartialShape{seq_begins});
    check_shape(this, InputIdx::BLOCK_INDICES_BEGINS, "block_indices_begins", ov::PartialShape{seq_begins});
    check_float_inputs(this, {InputIdx::QUERY, InputIdx::KEY, InputIdx::VALUE});

    // The page size is the block size of the caches (dim 2); key_block_size must divide it.
    for (auto idx : {InputIdx::KEY_CACHE, InputIdx::VALUE_CACHE}) {
        const auto& cache_ps = get_input_partial_shape(idx);
        NODE_VALIDATION_CHECK(this, cache_ps.rank().compatible(4), "SparsePA expects a rank-4 cache input ", idx, ", got ", cache_ps);
        if (cache_ps.rank().is_static() && cache_ps[2].is_static()) {
            const auto page_size = static_cast<size_t>(cache_ps[2].get_length());
            NODE_VALIDATION_CHECK(this,
                                  page_size % m_config.key_block_size == 0,
                                  "SparsePA: the cache page size ",
                                  page_size,
                                  " must be divisible by key_block_size ",
                                  m_config.key_block_size);
        }
    }

    const auto& scale_ps = get_input_partial_shape(InputIdx::SCALE);
    NODE_VALIDATION_CHECK(this,
                          scale_ps.is_dynamic() || ov::shape_size(scale_ps.to_shape()) == 1,
                          "SparsePA expects a scalar or 1-element scale input, got ",
                          scale_ps);
    const auto& sinks_ps = get_input_partial_shape(InputIdx::SINKS);
    NODE_VALIDATION_CHECK(this,
                          sinks_ps.rank().compatible(1) && (sinks_ps.rank().is_dynamic() || sinks_ps[0].compatible(0) ||
                                                              sinks_ps[0].compatible(static_cast<int64_t>(m_config.num_heads))),
                          "SparsePA expects an empty or [num_heads] sinks input, got ",
                          sinks_ps);

    auto out_ps = ov::PartialShape::dynamic(2);
    if (query_ps.rank().is_static()) {
        out_ps[0] = query_ps[0];
    }
    out_ps[1] = ov::Dimension(static_cast<int64_t>(m_config.num_heads * m_config.v_head_size));

    const auto out_et = m_output_type == ov::element::dynamic ? get_input_element_type(InputIdx::QUERY) : m_output_type;
    set_output_size(1);
    set_output_type(0, out_et, out_ps);
}

bool SparsePA::has_sinks() const {
    const auto& ps = get_input_partial_shape(InputIdx::SINKS);
    return !(ps.rank().is_static() && ps.size() == 1 && ps[0].is_static() && ps[0].get_length() == 0);
}

bool SparsePA::has_selected_block_mask() const {
    if (get_input_size() <= InputIdx::SELECTED_BLOCK_MASK) {
        return false;
    }
    const auto& ps = get_input_partial_shape(InputIdx::SELECTED_BLOCK_MASK);
    return !(ps.is_static() && ov::shape_size(ps.to_shape()) == 0);
}

std::shared_ptr<Node> SparsePA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SparsePA>(new_args, m_config, m_output_type);
}

}  // namespace ov::intel_gpu::op
