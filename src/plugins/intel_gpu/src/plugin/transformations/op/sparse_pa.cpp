// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sparse_pa.hpp"

#include <string>

#include "qsa_op_utils.hpp"

namespace ov::intel_gpu::op {

using namespace qsa_util;

SparsePA::SparsePA(const ov::OutputVector& args, const Config& config) : Base(args, static_cast<int64_t>(config.key_block_size)), m_config(config) {
    constructor_validate_and_infer_types();
}

bool SparsePA::validate_config(const Config& c, std::string& reason) {
    return validate_sparse_attention_config(c, reason);
}

bool SparsePA::has_sinks() const {
    const auto& ps = get_input_partial_shape(InputIdx::SINKS);
    return !ps.rank().is_static() || ps.size() != 1 || !ps[0].is_static() || ps[0].get_length() != 0;
}

bool SparsePA::visit_attributes(ov::AttributeVisitor& visitor) {
    Base::visit_attributes(visitor);
    visitor.on_attribute("num_heads", m_config.num_heads);
    visitor.on_attribute("num_kv_heads", m_config.num_kv_heads);
    visitor.on_attribute("num_selection_heads", m_config.num_selection_heads);
    visitor.on_attribute("k_head_size", m_config.k_head_size);
    visitor.on_attribute("v_head_size", m_config.v_head_size);
    visitor.on_attribute("query_block_size", m_config.query_block_size);
    visitor.on_attribute("causal", m_config.causal);
    return true;
}

void SparsePA::validate_and_infer_types() {
    Base::validate_and_infer_types();
    if (m_config.num_heads == 0) {
        return;  // default-constructed; the config is not set yet
    }
    std::string reason;
    NODE_VALIDATION_CHECK(this, validate_config(m_config, reason), "SparsePA configuration is invalid: ", reason);
    NODE_VALIDATION_CHECK(this,
                          m_config.key_block_size == static_cast<size_t>(get_key_block_size()),
                          "SparsePA key_block_size must match the PagedSparseAttention attribute");

    // The core op leaves the head geometry to the backend: tie the projections and caches to the config.
    auto width_matches = [this](size_t input, size_t expected) {
        const auto& ps = get_input_partial_shape(input);
        return ps.rank().is_dynamic() || ps.size() != 2 || ps[1].compatible(static_cast<int64_t>(expected));
    };
    NODE_VALIDATION_CHECK(this,
                          width_matches(InputIdx::QUERY, m_config.num_heads * m_config.k_head_size) &&
                              width_matches(InputIdx::KEY, m_config.num_kv_heads * m_config.k_head_size) &&
                              width_matches(InputIdx::VALUE, m_config.num_kv_heads * m_config.v_head_size),
                          "SparsePA query/key/value widths do not match num_heads, num_kv_heads and the head sizes");
    const auto& key_cache = get_input_partial_shape(InputIdx::KEY_CACHE);
    const auto& value_cache = get_input_partial_shape(InputIdx::VALUE_CACHE);
    NODE_VALIDATION_CHECK(
        this,
        (key_cache.rank().is_dynamic() || (key_cache.size() == 4 && key_cache[1].compatible(static_cast<int64_t>(m_config.num_kv_heads)) &&
                                           key_cache[3].compatible(static_cast<int64_t>(m_config.k_head_size)))) &&
            (value_cache.rank().is_dynamic() || (value_cache.size() == 4 && value_cache[1].compatible(static_cast<int64_t>(m_config.num_kv_heads)) &&
                                                 value_cache[3].compatible(static_cast<int64_t>(m_config.v_head_size)))),
        "SparsePA caches must be [blocks, num_kv_heads, page_size, head_size]");
}

std::shared_ptr<Node> SparsePA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SparsePA>(new_args, m_config);
}

}  // namespace ov::intel_gpu::op
