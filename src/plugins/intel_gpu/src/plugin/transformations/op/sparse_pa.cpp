// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/sparse_pa.hpp"

#include <string>
#include <utility>

#include "qsa_op_utils.hpp"

namespace ov::intel_gpu::op {

using namespace qsa_util;

namespace {

void visit_kv_quant(ov::AttributeVisitor& visitor, const std::string& prefix, KVQuant& quant) {
    auto mode = static_cast<int32_t>(quant.mode);
    auto bits = static_cast<int32_t>(quant.bits);
    auto group_size = static_cast<int32_t>(quant.group_size);
    visitor.on_attribute(prefix + "quant_mode", mode);
    visitor.on_attribute(prefix + "quant_bits", bits);
    visitor.on_attribute(prefix + "quant_symmetric", quant.symmetric);
    visitor.on_attribute(prefix + "quant_group_size", group_size);
    quant.mode = static_cast<KVQuant::Mode>(mode);
    quant.bits = static_cast<uint8_t>(bits);
    quant.group_size = static_cast<uint16_t>(group_size);
}

}  // namespace

SparsePA::SparsePA(const ov::OutputVector& args, const Config& config) : Base(args, static_cast<int64_t>(config.key_block_size)), m_config(config) {
    constructor_validate_and_infer_types();
}

bool SparsePA::validate_config(const Config& c, std::string& reason) {
    if (!validate_sparse_attention_config(c, reason)) {
        return false;
    }
    if (c.page_size != 0 && c.page_size % c.key_block_size != 0) {
        reason = "key_block_size must divide the page size";
        return false;
    }
    if (c.kv_quantized()) {
        if (!validate_kv_quant(c.key_quant, true, c.page_size, c.k_head_size, c.key_block_size, reason) ||
            !validate_kv_quant(c.value_quant, false, c.page_size, c.v_head_size, c.key_block_size, reason)) {
            return false;
        }
    }
    return true;
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
    visitor.on_attribute("page_size", m_config.page_size);
    visit_kv_quant(visitor, "key_", m_config.key_quant);
    visit_kv_quant(visitor, "value_", m_config.value_quant);
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
    // Quantized caches are u8 pages [blocks, Hkv, page_size + parameter rows, bytes per row].
    auto cache_matches = [this](const ov::PartialShape& ps, const KVQuant& quant, size_t head_size) {
        if (ps.rank().is_dynamic()) {
            return true;
        }
        if (ps.size() != 4 || !ps[1].compatible(static_cast<int64_t>(m_config.num_kv_heads))) {
            return false;
        }
        if (!quant.enabled()) {
            return ps[3].compatible(static_cast<int64_t>(head_size)) && (m_config.page_size == 0 || ps[2].compatible(static_cast<int64_t>(m_config.page_size)));
        }
        const auto layout = kv_page_layout(quant, m_config.page_size, head_size);
        return ps[2].compatible(static_cast<int64_t>(layout.total_rows)) && ps[3].compatible(static_cast<int64_t>(layout.row_bytes));
    };
    NODE_VALIDATION_CHECK(
        this,
        cache_matches(key_cache, m_config.key_quant, m_config.k_head_size) && cache_matches(value_cache, m_config.value_quant, m_config.v_head_size),
        "SparsePA caches must be [blocks, num_kv_heads, page_size, head_size] (quantized: [blocks, num_kv_heads, page_size + parameter rows, "
        "bytes per row])");
    for (auto [idx, quant] : {std::pair{InputIdx::KEY_CACHE, &m_config.key_quant}, std::pair{InputIdx::VALUE_CACHE, &m_config.value_quant}}) {
        const auto& type = get_input_element_type(idx);
        NODE_VALIDATION_CHECK(this, !quant->enabled() || type.is_dynamic() || type == ov::element::u8, "A quantized SparsePA cache must be u8, got ", type);
    }
}

std::shared_ptr<Node> SparsePA::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<SparsePA>(new_args, m_config);
}

}  // namespace ov::intel_gpu::op
