// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qsa_indexer.hpp"

#include <cmath>
#include <initializer_list>
#include <memory>
#include <string>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transformations/itt.hpp"

namespace ov::intel_cpu {

namespace {
// The kernels read every integer input as int32_t.
void check_i32_input(const ov::Node* node, size_t idx, const char* name, int64_t rank) {
    const auto& et = node->get_input_element_type(idx);
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node,
                          et.is_dynamic() || et == element::i32,
                          node->get_type_name(),
                          " expects an i32 ",
                          name,
                          " input, got ",
                          et);
    NODE_VALIDATION_CHECK(node,
                          ps.rank().compatible(rank),
                          node->get_type_name(),
                          " expects a rank-",
                          rank,
                          " ",
                          name,
                          " input, got ",
                          ps);
}

void check_shape(const ov::Node* node, size_t idx, const char* name, const ov::PartialShape& expected) {
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node,
                          ps.compatible(expected),
                          node->get_type_name(),
                          " expects ",
                          name,
                          " shaped ",
                          expected,
                          ", got ",
                          ps);
}

// Each port is validated independently, not merged into one shared type: the CPU kernel
// negotiates every floating-point port's precision separately (load_row_f32()/store_row_f32()
// convert per-element on access), so e.g. f32 compute ports alongside bf16 caches is a
// legitimate, intentional configuration, not a mismatch to reject.
void check_float_inputs(const ov::Node* node, std::initializer_list<size_t> indices) {
    for (auto idx : indices) {
        const auto& et = node->get_input_element_type(idx);
        NODE_VALIDATION_CHECK(node,
                              et.is_dynamic() || et.is_real(),
                              node->get_type_name(),
                              " expects a floating-point input at index ",
                              idx,
                              ", got ",
                              et);
    }
}

ov::Dimension dim_at(const ov::PartialShape& ps, size_t axis) {
    return ps.rank().is_static() && ps.size() > axis ? ps[axis] : ov::Dimension::dynamic();
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
    if (c.indexer_head_dim == 0) {
        return fail("indexer_head_dim must be positive");
    }
    if (c.indexer_rotary_dim % 2 != 0 || c.indexer_rotary_dim > c.indexer_head_dim) {
        return fail("indexer_rotary_dim must be even and not exceed indexer_head_dim");
    }
    if (!std::isfinite(c.scale) || c.scale < 0.0f) {
        return fail("scale must be positive (0 selects the default 1 / sqrt(indexer_head_dim))");
    }
    return true;
}

float QSAIndexer::effective_scale(const Config& c) {
    return c.scale > 0.0f ? c.scale : 1.0f / std::sqrt(static_cast<float>(c.indexer_head_dim));
}

size_t QSAIndexer::get_index_query_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::Q_IDX);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[2].is_static(),
                    "QSAIndexer: q_idx shape is not static");
    return static_cast<size_t>(ps[2].get_length());
}

size_t QSAIndexer::get_index_key_heads() const {
    const auto& ps = get_input_partial_shape(InputIdx::K_IDX);
    OPENVINO_ASSERT(ps.rank().is_static() && ps.size() == 4 && ps[2].is_static(),
                    "QSAIndexer: k_idx shape is not static");
    return static_cast<size_t>(ps[2].get_length());
}

bool QSAIndexer::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(QSAIndexer_visit_attributes);
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
    INTERNAL_OP_SCOPE(QSAIndexer_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == InputIdx::COUNT,
                          "QSAIndexer expects ",
                          InputIdx::COUNT,
                          " inputs, got ",
                          get_input_size());

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
    check_float_inputs(this,
                       {InputIdx::Q_IDX,
                        InputIdx::K_IDX,
                        InputIdx::ROPE_COS,
                        InputIdx::ROPE_SIN,
                        InputIdx::Q_NORM_WEIGHT,
                        InputIdx::K_NORM_WEIGHT,
                        InputIdx::SUMMARY_CACHE,
                        InputIdx::PAST_INCOMPLETE_RAW_K,
                        InputIdx::PAST_INCOMPLETE_ROPE});

    const auto r = static_cast<int64_t>(m_config.compress_ratio);
    const ov::Dimension di(static_cast<int64_t>(m_config.indexer_head_dim));
    const ov::Dimension rot(static_cast<int64_t>(m_config.indexer_rotary_dim));
    const ov::Dimension rope_tail(static_cast<int64_t>(2 * m_config.indexer_rotary_dim));
    const ov::Dimension tail_len(r - 1);
    const auto batch = dim_at(q_idx_ps, 0);
    const auto length = dim_at(q_idx_ps, 1);
    const auto h_ik = dim_at(k_idx_ps, 2);

    check_shape(this, InputIdx::Q_IDX, "q_idx", ov::PartialShape{batch, length, -1, di});
    check_shape(this, InputIdx::K_IDX, "k_idx", ov::PartialShape{batch, length, h_ik, di});
    check_shape(this, InputIdx::ROPE_COS, "rope_cos", ov::PartialShape{batch, length, rot});
    check_shape(this, InputIdx::ROPE_SIN, "rope_sin", ov::PartialShape{batch, length, rot});
    check_shape(this, InputIdx::Q_NORM_WEIGHT, "q_norm_weight", ov::PartialShape{di});
    check_shape(this, InputIdx::K_NORM_WEIGHT, "k_norm_weight", ov::PartialShape{di});

    check_i32_input(this, InputIdx::KEY_LENGTH, "key_length", 1);
    check_shape(this, InputIdx::KEY_LENGTH, "key_length", ov::PartialShape{batch});

    // Unlike QSAIndexer, summary_cache's block-capacity axis (Bp) has no single raw-buffer
    // capacity to independently derive it from: the per-row incomplete tail is a fixed-size
    // buffer, not a capacity-bearing cache. Bp is purely caller-managed (grown via Concat each
    // call, batch-max sized) -- only N/H_ik/D_idx are shape-consistency-checked here.
    check_shape(this, InputIdx::SUMMARY_CACHE, "summary_cache", ov::PartialShape{batch, h_ik, -1, di});
    check_shape(this,
                InputIdx::PAST_INCOMPLETE_RAW_K,
                "past_incomplete_raw_k",
                ov::PartialShape{batch, h_ik, tail_len, di});
    check_shape(this, InputIdx::PAST_INCOMPLETE_ROPE, "past_incomplete_rope", ov::PartialShape{batch, rope_tail});

    if (q_idx_ps.rank().is_static() && q_idx_ps.size() == 4 && q_idx_ps[2].is_static() && h_ik.is_static()) {
        const auto h_iq = static_cast<size_t>(q_idx_ps[2].get_length());
        const auto h_ik_value = static_cast<size_t>(h_ik.get_length());
        NODE_VALIDATION_CHECK(this,
                              h_ik_value != 0 && h_iq % h_ik_value == 0,
                              "QSAIndexer: H_iq must be a positive multiple of H_ik.");
        NODE_VALIDATION_CHECK(this,
                              m_config.h_sel == 1 || m_config.h_sel == h_ik_value,
                              "QSAIndexer: h_sel must be 1 or H_ik.");
    }

    const ov::PartialShape sel_indices_ps{batch,
                                          ov::Dimension(static_cast<int64_t>(m_config.h_sel)),
                                          length,
                                          ov::Dimension(static_cast<int64_t>(m_config.block_topk))};
    const ov::PartialShape sel_count_ps{batch, ov::Dimension(static_cast<int64_t>(m_config.h_sel)), length};

    set_output_size(OutputIdx::COUNT);
    set_output_type(OutputIdx::SEL_INDICES, element::i32, sel_indices_ps);
    set_output_type(OutputIdx::SEL_COUNT, element::i32, sel_count_ps);
    // The state updates are out-of-place: each state output mirrors its input.
    set_output_type(OutputIdx::SUMMARY_CACHE,
                    get_input_element_type(InputIdx::SUMMARY_CACHE),
                    get_input_partial_shape(InputIdx::SUMMARY_CACHE));
    set_output_type(OutputIdx::PAST_INCOMPLETE_RAW_K,
                    get_input_element_type(InputIdx::PAST_INCOMPLETE_RAW_K),
                    get_input_partial_shape(InputIdx::PAST_INCOMPLETE_RAW_K));
    set_output_type(OutputIdx::PAST_INCOMPLETE_ROPE,
                    get_input_element_type(InputIdx::PAST_INCOMPLETE_ROPE),
                    get_input_partial_shape(InputIdx::PAST_INCOMPLETE_ROPE));
}

std::shared_ptr<Node> QSAIndexer::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<QSAIndexer>(new_args, m_config);
}

}  // namespace ov::intel_cpu
