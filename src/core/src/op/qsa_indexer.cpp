// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"

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

QSAIndexer::QSAIndexer(const ov::OutputVector& args, const Config& config) : Op(args), m_config(config) {
    constructor_validate_and_infer_types();
}

bool QSAIndexer::validate_config(const Config& c, std::string& reason) {
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
    return true;
}

size_t QSAIndexer::get_index_query_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::Q_IDX);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[2].is_static(), "QSAIndexer: q_idx shape is not static");
    return static_cast<size_t>(ps[2].get_length());
}

size_t QSAIndexer::get_index_key_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::K_IDX);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[2].is_static(), "QSAIndexer: k_idx shape is not static");
    return static_cast<size_t>(ps[2].get_length());
}

bool QSAIndexer::visit_attributes(ov::AttributeVisitor& visitor) {
    OV_OP_SCOPE(internal_QSAIndexer_visit_attributes);
    visitor.on_attribute("compress_ratio", m_config.compress_ratio);
    visitor.on_attribute("block_topk", m_config.block_topk);
    visitor.on_attribute("h_sel", m_config.h_sel);
    visitor.on_attribute("indexer_rotary_dim", m_config.indexer_rotary_dim);
    visitor.on_attribute("indexer_head_dim", m_config.indexer_head_dim);
    visitor.on_attribute("eps", m_config.eps);
    visitor.on_attribute("scale", m_config.scale);
    return true;
}

void QSAIndexer::validate_and_infer_types() {
    OV_OP_SCOPE(internal_QSAIndexer_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == InputIdx::COUNT, "QSAIndexer expects ", InputIdx::COUNT, " inputs, got ", get_input_size());

    std::string reason;
    NODE_VALIDATION_CHECK(this, validate_config(m_config, reason), "QSAIndexer configuration is invalid: ", reason);

    const auto& q_idx_ps = get_input_partial_shape(InputIdx::Q_IDX);
    const auto& k_idx_ps = get_input_partial_shape(InputIdx::K_IDX);
    NODE_VALIDATION_CHECK(this,
                           q_idx_ps.rank().is_dynamic() || q_idx_ps.size() == 4,
                           "QSAIndexer expects a rank-4 [B, L, H_iq, D_idx] q_idx input, got ",
                           q_idx_ps);
    NODE_VALIDATION_CHECK(this,
                           k_idx_ps.rank().is_dynamic() || k_idx_ps.size() == 4,
                           "QSAIndexer expects a rank-4 [B, L, H_ik, D_idx] k_idx input, got ",
                           k_idx_ps);
    const auto& rotary_ps = get_input_partial_shape(InputIdx::ROTARY_COS_SIN);
    NODE_VALIDATION_CHECK(this,
                           rotary_ps.rank().is_dynamic() ||
                               (rotary_ps.size() == 2 && rotary_ps[1].compatible(static_cast<int64_t>(m_config.indexer_rotary_dim))),
                           "QSAIndexer expects a [max_positions, indexer_rotary_dim=",
                           m_config.indexer_rotary_dim,
                           "] rotary_cos_sin input, got ",
                           rotary_ps);

    ov::Dimension batch = ov::Dimension::dynamic();
    ov::Dimension length = ov::Dimension::dynamic();
    if (q_idx_ps.rank().is_static() && q_idx_ps.size() == 4) {
        batch = q_idx_ps[0];
        length = q_idx_ps[1];
    }

    check_i32_input(this, InputIdx::POSITION_IDS, "position_ids", 2);
    check_i32_input(this, InputIdx::PAST_LENS, "past_lens", 1);
    check_i32_input(this, InputIdx::INDEXER_POSITION_CACHE, "indexer_position_cache", 2);
    const auto& position_ids_ps = get_input_partial_shape(InputIdx::POSITION_IDS);
    NODE_VALIDATION_CHECK(this,
                           position_ids_ps.compatible(ov::PartialShape{batch, length}),
                           "QSAIndexer expects position_ids shaped [B, L] like q_idx, got ",
                           position_ids_ps);
    NODE_VALIDATION_CHECK(this,
                           get_input_partial_shape(InputIdx::PAST_LENS).compatible(ov::PartialShape{batch}),
                           "QSAIndexer expects past_lens shaped [B], got ",
                           get_input_partial_shape(InputIdx::PAST_LENS));

    if (q_idx_ps.rank().is_static() && q_idx_ps.size() == 4 && q_idx_ps[2].is_static() && k_idx_ps.rank().is_static() && k_idx_ps.size() == 4 &&
        k_idx_ps[2].is_static()) {
        const auto h_iq = static_cast<size_t>(q_idx_ps[2].get_length());
        const auto h_ik = static_cast<size_t>(k_idx_ps[2].get_length());
        NODE_VALIDATION_CHECK(this, h_ik != 0 && h_iq % h_ik == 0, "QSAIndexer: H_iq must be a positive multiple of H_ik.");
        NODE_VALIDATION_CHECK(this, m_config.h_sel == 1 || m_config.h_sel == h_ik, "QSAIndexer: h_sel must be 1 or H_ik.");
    }

    const ov::PartialShape sel_indices_ps{batch, ov::Dimension(static_cast<int64_t>(m_config.h_sel)), length,
                                           ov::Dimension(static_cast<int64_t>(m_config.block_topk))};
    const ov::PartialShape sel_count_ps{batch, ov::Dimension(static_cast<int64_t>(m_config.h_sel)), length};

    set_output_size(2);
    set_output_type(0, element::i32, sel_indices_ps);
    set_output_type(1, element::i32, sel_count_ps);
}

std::shared_ptr<Node> QSAIndexer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<QSAIndexer>(new_args, m_config);
}

}  // namespace ov::op::internal
