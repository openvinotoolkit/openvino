// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"

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
    if (c.indexer_head_dim == 0 || c.indexer_head_dim % 16 != 0) {
        return fail("indexer_head_dim must be a positive multiple of 16");
    }
    if (c.indexer_rotary_dim % 2 != 0 || c.indexer_rotary_dim > c.indexer_head_dim) {
        return fail("indexer_rotary_dim must be even and not exceed indexer_head_dim");
    }
    if (c.page_size == 0 || c.page_size % c.compress_ratio != 0) {
        return fail("page_size must be positive and divisible by compress_ratio");
    }
    return true;
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
    const auto& rotary_ps = get_input_partial_shape(InputIdx::ROTARY_COS_SIN);
    NODE_VALIDATION_CHECK(this,
                           rotary_ps.rank().is_dynamic() ||
                               (rotary_ps.size() == 2 && rotary_ps[1].compatible(static_cast<int64_t>(m_config.indexer_rotary_dim))),
                           "PagedQSAIndexer expects a [max_positions, indexer_rotary_dim=",
                           m_config.indexer_rotary_dim,
                           "] rotary_cos_sin input, got ",
                           rotary_ps);

    ov::Dimension tokens = ov::Dimension::dynamic();
    if (q_idx_ps.rank().is_static() && q_idx_ps.size() == 2) {
        tokens = q_idx_ps[0];
    }

    check_i32_input(this, InputIdx::POSITION_IDS, "position_ids", 1);
    check_i32_input(this, InputIdx::BLOCK_INDICES, "block_indices", 1);
    check_i32_input(this, InputIdx::BLOCK_INDICES_BEGINS, "block_indices_begins", 1);
    check_i32_input(this, InputIdx::SUBSEQUENCE_BEGINS, "subsequence_begins", 1);
    check_i32_input(this, InputIdx::PAST_LENS, "past_lens", 1);
    check_i32_input(this, InputIdx::INDEXER_POSITION_CACHE, "indexer_position_cache", 2);
    NODE_VALIDATION_CHECK(this,
                           get_input_partial_shape(InputIdx::POSITION_IDS).compatible(ov::PartialShape{tokens}),
                           "PagedQSAIndexer expects position_ids shaped [tokens] like q_idx, got ",
                           get_input_partial_shape(InputIdx::POSITION_IDS));

    if (q_idx_ps.rank().is_static() && q_idx_ps.size() == 2 && q_idx_ps[1].is_static() && k_idx_ps.rank().is_static() && k_idx_ps.size() == 2 &&
        k_idx_ps[1].is_static()) {
        const auto h_iq = static_cast<size_t>(q_idx_ps[1].get_length()) / m_config.indexer_head_dim;
        const auto h_ik = static_cast<size_t>(k_idx_ps[1].get_length()) / m_config.indexer_head_dim;
        NODE_VALIDATION_CHECK(this, h_ik != 0 && h_iq % h_ik == 0, "PagedQSAIndexer: H_iq must be a positive multiple of H_ik.");
        NODE_VALIDATION_CHECK(this, m_config.h_sel == 1 || m_config.h_sel == h_ik, "PagedQSAIndexer: h_sel must be 1 or H_ik.");
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
