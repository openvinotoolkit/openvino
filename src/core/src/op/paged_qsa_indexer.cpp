// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"

#include <cmath>
#include <initializer_list>
#include <string>

#include "itt.hpp"

namespace ov::op::internal {

namespace {
// The kernels read every integer input as int32_t.
void check_i32_input(const ov::Node* node, size_t idx, const char* name, int64_t rank) {
    const auto& et = node->get_input_element_type(idx);
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node, et.is_dynamic() || et == element::i32, node->get_type_name(), " expects an i32 ", name, " input, got ", et);
    NODE_VALIDATION_CHECK(node, ps.rank().compatible(rank), node->get_type_name(), " expects a rank-", rank, " ", name, " input, got ", ps);
}

void check_shape(const ov::Node* node, size_t idx, const char* name, const ov::PartialShape& expected) {
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node, ps.compatible(expected), node->get_type_name(), " expects ", name, " shaped ", expected, ", got ", ps);
}

void check_float_inputs(const ov::Node* node, std::initializer_list<size_t> indices) {
    element::Type et = element::dynamic;
    for (auto idx : indices) {
        const auto& in_et = node->get_input_element_type(idx);
        NODE_VALIDATION_CHECK(node,
                              element::Type::merge(et, et, in_et),
                              node->get_type_name(),
                              " expects all floating-point inputs to share one element type; input ",
                              idx,
                              " is ",
                              in_et);
    }
    NODE_VALIDATION_CHECK(node, et.is_dynamic() || et.is_real(), node->get_type_name(), " expects floating-point inputs, got ", et);
}

ov::Dimension dim_at(const ov::PartialShape& ps, size_t axis) {
    return ps.rank().is_static() && ps.size() > axis ? ps[axis] : ov::Dimension::dynamic();
}

/// Head count of a flattened [tokens, H * D_idx] input; dynamic unless the last axis is static.
ov::Dimension flat_heads(const ov::Node* node, const ov::PartialShape& ps, size_t head_dim, const char* name) {
    const auto last = dim_at(ps, 1);
    if (last.is_dynamic()) {
        return ov::Dimension::dynamic();
    }
    NODE_VALIDATION_CHECK(node,
                          last.get_length() > 0 && last.get_length() % static_cast<int64_t>(head_dim) == 0,
                          node->get_type_name(),
                          " expects the ",
                          name,
                          " last dimension to be a positive multiple of indexer_head_dim=",
                          head_dim,
                          ", got ",
                          ps);
    return last.get_length() / static_cast<int64_t>(head_dim);
}
}  // namespace

PagedQSAIndexer::PagedQSAIndexer(const ov::OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

bool PagedQSAIndexer::validate_config(const Config& c, std::string& reason) {
    auto fail = [&reason](const char* msg) {
        reason = msg;
        return false;
    };
    if (c.compress_ratio == 0) {
        return fail("compress_ratio must be positive");
    }
    if (c.block_topk == 0) {
        return fail("block_topk must be positive");
    }
    if (c.h_sel == 0) {
        return fail("h_sel must be positive");
    }
    if (c.indexer_head_dim == 0) {
        return fail("indexer_head_dim must be positive");
    }
    if (c.indexer_rotary_dim % 2 != 0 || c.indexer_rotary_dim > c.indexer_head_dim) {
        return fail("indexer_rotary_dim must be even and not exceed indexer_head_dim");
    }
    if (c.page_size == 0 || c.page_size % c.compress_ratio != 0) {
        return fail("page_size must be positive and divisible by compress_ratio");
    }
    if (!std::isfinite(c.scale) || c.scale < 0.0f) {
        return fail("scale must be positive (0 selects the default 1 / sqrt(indexer_head_dim))");
    }
    return true;
}

float PagedQSAIndexer::effective_scale(const Config& c) {
    return c.scale > 0.0f ? c.scale : 1.0f / std::sqrt(static_cast<float>(c.indexer_head_dim));
}

size_t PagedQSAIndexer::get_index_query_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::Q_IDX);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 2 && ps[1].is_static(), "PagedQSAIndexer: q_idx shape is not static");
    OPENVINO_ASSERT(m_config.indexer_head_dim != 0, "PagedQSAIndexer: indexer_head_dim is 0");
    return static_cast<size_t>(ps[1].get_length()) / m_config.indexer_head_dim;
}

size_t PagedQSAIndexer::get_index_key_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::K_IDX);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 2 && ps[1].is_static(), "PagedQSAIndexer: k_idx shape is not static");
    OPENVINO_ASSERT(m_config.indexer_head_dim != 0, "PagedQSAIndexer: indexer_head_dim is 0");
    return static_cast<size_t>(ps[1].get_length()) / m_config.indexer_head_dim;
}

bool PagedQSAIndexer::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_PagedQSAIndexer_visit_attributes);
    visitor.on_attribute("compress_ratio", m_config.compress_ratio);
    visitor.on_attribute("block_topk", m_config.block_topk);
    visitor.on_attribute("h_sel", m_config.h_sel);
    visitor.on_attribute("indexer_rotary_dim", m_config.indexer_rotary_dim);
    visitor.on_attribute("indexer_head_dim", m_config.indexer_head_dim);
    visitor.on_attribute("page_size", m_config.page_size);
    visitor.on_attribute("eps", m_config.eps);
    visitor.on_attribute("scale", m_config.scale);
    return true;
}

void PagedQSAIndexer::validate_and_infer_types() {
    OV_OP_SCOPE(internal_PagedQSAIndexer_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == InputIdx::COUNT, "PagedQSAIndexer expects ", InputIdx::COUNT, " inputs, got ", get_input_size());

    std::string reason;
    NODE_VALIDATION_CHECK(this, validate_config(m_config, reason), "PagedQSAIndexer configuration is invalid: ", reason);
    // Resolve the 0 sentinel so serialized IR always carries the actual multiplier.
    m_config.scale = effective_scale(m_config);

    const auto& q_idx_ps = get_input_partial_shape(InputIdx::Q_IDX);
    const auto& k_idx_ps = get_input_partial_shape(InputIdx::K_IDX);
    NODE_VALIDATION_CHECK(this,
                           q_idx_ps.rank().is_dynamic() || q_idx_ps.size() == 2,
                           "PagedQSAIndexer expects a rank-2 [tokens, H_iq*D_idx] q_idx input, got ",
                           q_idx_ps);
    NODE_VALIDATION_CHECK(this,
                           k_idx_ps.rank().is_dynamic() || k_idx_ps.size() == 2,
                           "PagedQSAIndexer expects a rank-2 [tokens, H_ik*D_idx] k_idx input, got ",
                           k_idx_ps);
    check_float_inputs(this,
                       {InputIdx::Q_IDX,
                        InputIdx::K_IDX,
                        InputIdx::ROPE_COS,
                        InputIdx::ROPE_SIN,
                        InputIdx::SUMMARY_CACHE,
                        InputIdx::INDEXER_RAW_K_CACHE,
                        InputIdx::Q_NORM_WEIGHT,
                        InputIdx::K_NORM_WEIGHT,
                        InputIdx::INDEXER_ROPE_CACHE});

    const ov::Dimension tokens = dim_at(q_idx_ps, 0);
    const ov::Dimension di(static_cast<int64_t>(m_config.indexer_head_dim));
    const ov::Dimension rot(static_cast<int64_t>(m_config.indexer_rotary_dim));
    const ov::Dimension rope_row(static_cast<int64_t>(2 * m_config.indexer_rotary_dim));
    const ov::Dimension page(static_cast<int64_t>(m_config.page_size));
    const ov::Dimension spp(static_cast<int64_t>(m_config.page_size / m_config.compress_ratio));
    const auto h_iq = flat_heads(this, q_idx_ps, m_config.indexer_head_dim, "q_idx");
    const auto h_ik = flat_heads(this, k_idx_ps, m_config.indexer_head_dim, "k_idx");

    check_shape(this, InputIdx::K_IDX, "k_idx", ov::PartialShape{tokens, -1});
    check_shape(this, InputIdx::ROPE_COS, "rope_cos", ov::PartialShape{tokens, rot});
    check_shape(this, InputIdx::ROPE_SIN, "rope_sin", ov::PartialShape{tokens, rot});
    check_shape(this, InputIdx::Q_NORM_WEIGHT, "q_norm_weight", ov::PartialShape{di});
    check_shape(this, InputIdx::K_NORM_WEIGHT, "k_norm_weight", ov::PartialShape{di});

    // The three paged caches share the main KV page pool (dim 0).
    ov::Dimension num_blocks = ov::Dimension::dynamic();
    for (auto idx : {InputIdx::SUMMARY_CACHE, InputIdx::INDEXER_RAW_K_CACHE, InputIdx::INDEXER_ROPE_CACHE}) {
        NODE_VALIDATION_CHECK(this,
                              ov::Dimension::merge(num_blocks, num_blocks, dim_at(get_input_partial_shape(idx), 0)),
                              "PagedQSAIndexer expects summary_cache, indexer_raw_k_cache and indexer_rope_cache to share dim 0 (num_blocks)");
    }
    check_shape(this, InputIdx::SUMMARY_CACHE, "summary_cache", ov::PartialShape{num_blocks, h_ik, spp, di});
    check_shape(this, InputIdx::INDEXER_RAW_K_CACHE, "indexer_raw_k_cache", ov::PartialShape{num_blocks, h_ik, page, di});
    check_shape(this, InputIdx::INDEXER_ROPE_CACHE, "indexer_rope_cache", ov::PartialShape{num_blocks, spp, rope_row});

    check_i32_input(this, InputIdx::BLOCK_INDICES, "block_indices", 1);
    check_i32_input(this, InputIdx::BLOCK_INDICES_BEGINS, "block_indices_begins", 1);
    check_i32_input(this, InputIdx::SUBSEQUENCE_BEGINS, "subsequence_begins", 1);
    check_i32_input(this, InputIdx::PAST_LENS, "past_lens", 1);
    const auto num_seqs = dim_at(get_input_partial_shape(InputIdx::PAST_LENS), 0);
    const auto num_seqs_plus_one = num_seqs.is_static() ? ov::Dimension(num_seqs.get_length() + 1) : ov::Dimension::dynamic();
    check_shape(this, InputIdx::SUBSEQUENCE_BEGINS, "subsequence_begins", ov::PartialShape{num_seqs_plus_one});
    check_shape(this, InputIdx::BLOCK_INDICES_BEGINS, "block_indices_begins", ov::PartialShape{num_seqs_plus_one});

    if (h_iq.is_static() && h_ik.is_static()) {
        const auto h_iq_value = static_cast<size_t>(h_iq.get_length());
        const auto h_ik_value = static_cast<size_t>(h_ik.get_length());
        NODE_VALIDATION_CHECK(this, h_iq_value % h_ik_value == 0, "PagedQSAIndexer: H_iq must be a positive multiple of H_ik.");
        NODE_VALIDATION_CHECK(this, m_config.h_sel == 1 || m_config.h_sel == h_ik_value, "PagedQSAIndexer: h_sel must be 1 or H_ik.");
    }

    const ov::PartialShape sel_indices_ps{tokens, ov::Dimension(static_cast<int64_t>(m_config.h_sel)), ov::Dimension(static_cast<int64_t>(m_config.block_topk))};
    const ov::PartialShape sel_count_ps{tokens, ov::Dimension(static_cast<int64_t>(m_config.h_sel))};

    set_output_size(2);
    set_output_type(0, element::i32, sel_indices_ps);
    set_output_type(1, element::i32, sel_count_ps);
}

std::shared_ptr<Node> PagedQSAIndexer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<PagedQSAIndexer>(new_args, m_config);
}

}  // namespace ov::op::internal
