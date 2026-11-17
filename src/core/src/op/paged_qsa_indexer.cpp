// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"

#include <cmath>
#include <initializer_list>
#include <string>

#include "itt.hpp"

namespace ov::op::internal {

PagedQSAIndexer::PagedQSAIndexer(const ov::OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

bool PagedQSAIndexer::validate_config(const Config& c, std::string& reason) {
    if (c.key_block_size == 0) {
        reason = "key_block_size must be positive";
        return false;
    }
    if (c.block_topk == 0) {
        reason = "block_topk must be positive";
        return false;
    }
    if (c.num_selection_heads == 0) {
        reason = "num_selection_heads must be positive";
        return false;
    }
    if (c.indexer_head_dim == 0) {
        reason = "indexer_head_dim must be positive";
        return false;
    }
    if (c.indexer_rotary_dim % 2 != 0 || c.indexer_rotary_dim > c.indexer_head_dim) {
        reason = "indexer_rotary_dim must be even and not exceed indexer_head_dim";
        return false;
    }
    if (!std::isfinite(c.scale) || c.scale < 0.0f) {
        reason = "scale must be positive (0 selects the default 1 / sqrt(indexer_head_dim))";
        return false;
    }
    return true;
}

float PagedQSAIndexer::effective_scale(const Config& c) {
    return c.scale > 0.0f ? c.scale : 1.0f / std::sqrt(static_cast<float>(c.indexer_head_dim));
}

size_t PagedQSAIndexer::get_index_query_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::INDEX_QUERY);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 2 && ps[1].is_static(),
                    "PagedQSAIndexer: index_query shape is not static");
    OPENVINO_ASSERT(m_config.indexer_head_dim != 0, "PagedQSAIndexer: indexer_head_dim is 0");
    return static_cast<size_t>(ps[1].get_length()) / m_config.indexer_head_dim;
}

size_t PagedQSAIndexer::get_index_key_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::RAW_INDEX_KEYS);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 2 && ps[1].is_static(),
                    "PagedQSAIndexer: raw_index_keys shape is not static");
    OPENVINO_ASSERT(m_config.indexer_head_dim != 0, "PagedQSAIndexer: indexer_head_dim is 0");
    return static_cast<size_t>(ps[1].get_length()) / m_config.indexer_head_dim;
}

bool PagedQSAIndexer::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_PagedQSAIndexer_visit_attributes);
    visitor.on_attribute("key_block_size", m_config.key_block_size);
    visitor.on_attribute("block_topk", m_config.block_topk);
    visitor.on_attribute("num_selection_heads", m_config.num_selection_heads);
    visitor.on_attribute("indexer_rotary_dim", m_config.indexer_rotary_dim);
    visitor.on_attribute("indexer_head_dim", m_config.indexer_head_dim);
    visitor.on_attribute("epsilon", m_config.epsilon);
    visitor.on_attribute("scale", m_config.scale);
    return true;
}

void PagedQSAIndexer::validate_and_infer_types() {
    OV_OP_SCOPE(internal_PagedQSAIndexer_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == InputIdx::COUNT,
                          "PagedQSAIndexer expects ",
                          InputIdx::COUNT,
                          " inputs, got ",
                          get_input_size());

    std::string reason;
    NODE_VALIDATION_CHECK(this,
                          validate_config(m_config, reason),
                          "PagedQSAIndexer configuration is invalid: ",
                          reason);

    const auto& index_query_ps = get_input_partial_shape(InputIdx::INDEX_QUERY);
    const auto& raw_index_keys_ps = get_input_partial_shape(InputIdx::RAW_INDEX_KEYS);
    NODE_VALIDATION_CHECK(this,
                          index_query_ps.rank().is_dynamic() || index_query_ps.size() == 2,
                          "PagedQSAIndexer expects a rank-2 [tokens, H_iq*D_idx] index_query input, got ",
                          index_query_ps);
    NODE_VALIDATION_CHECK(this,
                          raw_index_keys_ps.rank().is_dynamic() || raw_index_keys_ps.size() == 2,
                          "PagedQSAIndexer expects a rank-2 [tokens, H_ik*D_idx] raw_index_keys input, got ",
                          raw_index_keys_ps);
    // All floating-point inputs must share one element type.
    element::Type float_et = element::dynamic;
    for (const auto idx : {InputIdx::INDEX_QUERY,
                           InputIdx::RAW_INDEX_KEYS,
                           InputIdx::ROPE_COS,
                           InputIdx::ROPE_SIN,
                           InputIdx::BLOCK_INDEX_KEYS_CACHE,
                           InputIdx::INDEX_KEY_CACHE,
                           InputIdx::Q_NORM_WEIGHT,
                           InputIdx::K_NORM_WEIGHT,
                           InputIdx::INDEX_BLOCK_ROPE_CACHE}) {
        const auto& in_et = get_input_element_type(idx);
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(float_et, float_et, in_et),
                              "PagedQSAIndexer expects all floating-point inputs to share one element type; input ",
                              idx,
                              " is ",
                              in_et);
    }
    NODE_VALIDATION_CHECK(this,
                          float_et.is_dynamic() || float_et.is_real(),
                          "PagedQSAIndexer expects floating-point inputs, got ",
                          float_et);

    // The kernels read every integer input as int32_t.
    for (const auto idx :
         {InputIdx::PAST_LENS, InputIdx::SUBSEQUENCE_BEGINS, InputIdx::BLOCK_INDICES, InputIdx::BLOCK_INDICES_BEGINS}) {
        const auto& in_et = get_input_element_type(idx);
        const auto& in_ps = get_input_partial_shape(idx);
        NODE_VALIDATION_CHECK(this,
                              in_et.is_dynamic() || in_et == element::i32,
                              "PagedQSAIndexer expects an i32 index input; input ",
                              idx,
                              " is ",
                              in_et);
        NODE_VALIDATION_CHECK(this,
                              in_ps.rank().is_dynamic() || in_ps.size() == 1,
                              "PagedQSAIndexer expects a rank-1 index input; input ",
                              idx,
                              " is ",
                              in_ps);
    }

    // Local helper: the extent of one axis, or dynamic when the rank or the axis is unknown.
    const auto dim_at = [](const ov::PartialShape& ps, size_t axis) -> ov::Dimension {
        return ps.rank().is_static() && ps.size() > axis ? ps[axis] : ov::Dimension::dynamic();
    };

    const ov::Dimension tokens = dim_at(index_query_ps, 0);
    const ov::Dimension di(static_cast<int64_t>(m_config.indexer_head_dim));
    const ov::Dimension rot(static_cast<int64_t>(m_config.indexer_rotary_dim));
    const ov::Dimension rope_row(static_cast<int64_t>(2 * m_config.indexer_rotary_dim));
    // page_size is not an attribute: it is the block size of the raw-K cache (dim 2).
    const ov::Dimension page = dim_at(get_input_partial_shape(InputIdx::INDEX_KEY_CACHE), 2);
    ov::Dimension spp = ov::Dimension::dynamic();
    if (page.is_static()) {
        const auto page_size = static_cast<size_t>(page.get_length());
        NODE_VALIDATION_CHECK(
            this,
            page_size > 0 && page_size % m_config.key_block_size == 0,
            "PagedQSAIndexer: index_key_cache dim 2 (page_size) must be a positive multiple of key_block_size, got ",
            page_size);
        spp = ov::Dimension(static_cast<int64_t>(page_size / m_config.key_block_size));
    }
    // H_iq / H_ik come from the flattened [tokens, H * D_idx] inputs; dynamic unless the last axis is static.
    const auto flat_heads = [this, &dim_at](const ov::PartialShape& ps, const char* name) -> ov::Dimension {
        const auto last = dim_at(ps, 1);
        if (last.is_dynamic()) {
            return ov::Dimension::dynamic();
        }
        NODE_VALIDATION_CHECK(
            this,
            last.get_length() > 0 && last.get_length() % static_cast<int64_t>(m_config.indexer_head_dim) == 0,
            "PagedQSAIndexer expects the ",
            name,
            " last dimension to be a positive multiple of indexer_head_dim=",
            m_config.indexer_head_dim,
            ", got ",
            ps);
        return ov::Dimension(last.get_length() / static_cast<int64_t>(m_config.indexer_head_dim));
    };
    const auto h_iq = flat_heads(index_query_ps, "index_query");
    const auto h_ik = flat_heads(raw_index_keys_ps, "raw_index_keys");

    NODE_VALIDATION_CHECK(this,
                          raw_index_keys_ps.compatible(ov::PartialShape{tokens, -1}),
                          "PagedQSAIndexer expects raw_index_keys shaped [tokens, H_ik*D_idx], got ",
                          raw_index_keys_ps);
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::ROPE_COS).compatible(ov::PartialShape{tokens, rot}),
                          "PagedQSAIndexer expects rope_cos shaped [tokens, indexer_rotary_dim], got ",
                          get_input_partial_shape(InputIdx::ROPE_COS));
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::ROPE_SIN).compatible(ov::PartialShape{tokens, rot}),
                          "PagedQSAIndexer expects rope_sin shaped [tokens, indexer_rotary_dim], got ",
                          get_input_partial_shape(InputIdx::ROPE_SIN));
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::Q_NORM_WEIGHT).compatible(ov::PartialShape{di}),
                          "PagedQSAIndexer expects q_norm_weight shaped [indexer_head_dim], got ",
                          get_input_partial_shape(InputIdx::Q_NORM_WEIGHT));
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::K_NORM_WEIGHT).compatible(ov::PartialShape{di}),
                          "PagedQSAIndexer expects k_norm_weight shaped [indexer_head_dim], got ",
                          get_input_partial_shape(InputIdx::K_NORM_WEIGHT));

    // The three paged caches share the main KV page pool (dim 0).
    ov::Dimension num_blocks = ov::Dimension::dynamic();
    for (auto idx : {InputIdx::BLOCK_INDEX_KEYS_CACHE, InputIdx::INDEX_KEY_CACHE, InputIdx::INDEX_BLOCK_ROPE_CACHE}) {
        NODE_VALIDATION_CHECK(this,
                              ov::Dimension::merge(num_blocks, num_blocks, dim_at(get_input_partial_shape(idx), 0)),
                              "PagedQSAIndexer expects block_index_keys_cache, index_key_cache and "
                              "index_block_rope_cache to share dim 0 (num_blocks)");
    }
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::BLOCK_INDEX_KEYS_CACHE)
                              .compatible(ov::PartialShape{num_blocks, h_ik, spp, di}),
                          "PagedQSAIndexer expects block_index_keys_cache shaped "
                          "[num_blocks, H_ik, page_size/r, indexer_head_dim], got ",
                          get_input_partial_shape(InputIdx::BLOCK_INDEX_KEYS_CACHE));
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(InputIdx::INDEX_KEY_CACHE).compatible(ov::PartialShape{num_blocks, h_ik, page, di}),
        "PagedQSAIndexer expects index_key_cache shaped [num_blocks, H_ik, page_size, indexer_head_dim], "
        "got ",
        get_input_partial_shape(InputIdx::INDEX_KEY_CACHE));
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(InputIdx::INDEX_BLOCK_ROPE_CACHE)
                              .compatible(ov::PartialShape{num_blocks, spp, rope_row}),
                          "PagedQSAIndexer expects index_block_rope_cache shaped "
                          "[num_blocks, page_size/r, 2*indexer_rotary_dim], got ",
                          get_input_partial_shape(InputIdx::INDEX_BLOCK_ROPE_CACHE));

    const auto num_seqs = dim_at(get_input_partial_shape(InputIdx::PAST_LENS), 0);
    const auto num_seqs_plus_one =
        num_seqs.is_static() ? ov::Dimension(num_seqs.get_length() + 1) : ov::Dimension::dynamic();
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(InputIdx::SUBSEQUENCE_BEGINS).compatible(ov::PartialShape{num_seqs_plus_one}),
        "PagedQSAIndexer expects subsequence_begins shaped [num_sequences+1], got ",
        get_input_partial_shape(InputIdx::SUBSEQUENCE_BEGINS));
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(InputIdx::BLOCK_INDICES_BEGINS).compatible(ov::PartialShape{num_seqs_plus_one}),
        "PagedQSAIndexer expects block_indices_begins shaped [num_sequences+1], got ",
        get_input_partial_shape(InputIdx::BLOCK_INDICES_BEGINS));

    if (h_iq.is_static() && h_ik.is_static()) {
        const auto h_iq_value = static_cast<size_t>(h_iq.get_length());
        const auto h_ik_value = static_cast<size_t>(h_ik.get_length());
        NODE_VALIDATION_CHECK(this,
                              h_iq_value % h_ik_value == 0,
                              "PagedQSAIndexer: H_iq must be a positive multiple of H_ik.");
        NODE_VALIDATION_CHECK(this,
                              m_config.num_selection_heads == 1 || m_config.num_selection_heads == h_ik_value,
                              "PagedQSAIndexer: num_selection_heads must be 1 or H_ik.");
    }

    const ov::PartialShape selected_block_indices_ps{tokens,
                                                     ov::Dimension(static_cast<int64_t>(m_config.num_selection_heads)),
                                                     ov::Dimension(static_cast<int64_t>(m_config.block_topk))};
    const ov::PartialShape selected_block_count_ps{tokens,
                                                   ov::Dimension(static_cast<int64_t>(m_config.num_selection_heads))};

    set_output_size(2);
    set_output_type(0, element::i32, selected_block_indices_ps);
    set_output_type(1, element::i32, selected_block_count_ps);
}

std::shared_ptr<Node> PagedQSAIndexer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<PagedQSAIndexer>(new_args, m_config);
}

}  // namespace ov::op::internal
