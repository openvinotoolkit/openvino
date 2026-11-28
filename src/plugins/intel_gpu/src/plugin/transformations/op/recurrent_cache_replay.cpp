// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/recurrent_cache_replay.hpp"

#include <cmath>
#include "openvino/core/validation_util.hpp"

namespace ov::intel_gpu::op {
namespace {
void floating(const Node* op, size_t port, bool bank = false) {
    const auto type = op->get_input_element_type(port);
    NODE_VALIDATION_CHECK(op, type == element::f32 || (!bank && type == element::f16),
                          "Replay requires f16/f32 data and FP32 banks");
    NODE_VALIDATION_CHECK(op, op->get_input_partial_shape(port).rank() == 4, "Replay data requires rank four");
}
Dimension bank_features(const Node* op, size_t port) {
    floating(op, port, true);
    const auto shape = op->get_input_partial_shape(port);
    NODE_VALIDATION_CHECK(op, shape[0].compatible(1) && shape[1].compatible(1), "Banks require [1,1,S,F]");
    return shape[3];
}
Dimension metadata_count(const Node* op, size_t port, bool pairs = false) {
    const auto type = op->get_input_element_type(port);
    NODE_VALIDATION_CHECK(op, type == element::i32 || type == element::i64, "Replay metadata requires i32/i64");
    const auto shape = op->get_input_partial_shape(port);
    const auto rank = shape.rank();
    NODE_VALIDATION_CHECK(op, rank == (pairs ? 2 : 1) || rank == 4, "Invalid replay metadata rank");
    if (rank == 4) {
        NODE_VALIDATION_CHECK(op, shape[0].compatible(1) && shape[1].compatible(1),
                              "Metadata leading dimensions must be one");
        if (pairs) {
            NODE_VALIDATION_CHECK(op, shape[3].compatible(2), "Row map requires pairs");
            return shape[2];
        }
        NODE_VALIDATION_CHECK(op, shape[2].compatible(1), "Vector metadata requires [1,1,1,N]");
        return shape[3];
    }
    if (pairs)
        NODE_VALIDATION_CHECK(op, shape[1].compatible(2), "Row map requires [N,2]");
    return shape[0];
}
void controls(const Node* op, size_t rows, size_t replay, size_t reset, const Dimension& batch) {
    const auto count = metadata_count(op, rows, true);
    const auto control = metadata_count(op, replay);
    metadata_count(op, reset);
    NODE_VALIDATION_CHECK(op, control.compatible(batch + 1), "Replay control contains B counts and keep");
    if (count.is_static() && batch.is_static())
        NODE_VALIDATION_CHECK(op, count.get_length() >= batch.get_length(),
                              "Row map must contain every active sequence");
}
} // namespace

RecurrentConvCacheReplayUpdate::RecurrentConvCacheReplayUpdate(const OutputVector& args, int64_t history)
    : Op(args), m_history(history) {
    constructor_validate_and_infer_types();
}
void RecurrentConvCacheReplayUpdate::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 6 && m_history > 0,
                          "Convolution replay requires six inputs and positive history");
    const auto features = bank_features(this, 0);
    floating(this, 2);
    floating(this, 3);
    const auto x = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(this, x[0].compatible(1), "Convolution input requires [1,B,T,C]");
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(3).compatible(PartialShape{1, 1, x[3], m_history + 1}),
                          "Invalid depthwise weights");
    if (features.is_static() && x[3].is_static()) {
        const auto channels = x[3].get_length();
        NODE_VALIDATION_CHECK(
            this, channels > 0 && features.get_length() % channels == 0 && features.get_length() / channels > m_history,
            "History bank must contain W committed and positive P pending columns");
    }
    controls(this, 1, 4, 5, x[1]);
    set_output_type(0, get_input_element_type(2), x);
}
bool RecurrentConvCacheReplayUpdate::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("history", m_history);
    return true;
}
std::shared_ptr<Node> RecurrentConvCacheReplayUpdate::clone_with_new_inputs(const OutputVector& args) const {
    check_new_args_count(this, args);
    return std::make_shared<RecurrentConvCacheReplayUpdate>(args, m_history);
}

RecurrentGatedDeltaNetReplayUpdate::RecurrentGatedDeltaNetReplayUpdate(const OutputVector& args, bool norm, float qe,
                                                                       float ke, std::string mapping)
    : Op(args), m_fuse_qk_l2norm(norm), m_q_l2_norm_eps(qe), m_k_l2_norm_eps(ke), m_head_mapping(std::move(mapping)) {
    constructor_validate_and_infer_types();
}
void RecurrentGatedDeltaNetReplayUpdate::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 12, "GDN replay requires twelve inputs");
    for (size_t port : {0, 1, 2, 4, 5})
        floating(this, port);
    const auto q = get_input_partial_shape(0), k = get_input_partial_shape(1), v = get_input_partial_shape(2),
               g = get_input_partial_shape(4);
    NODE_VALIDATION_CHECK(this,
                          q[0].compatible(v[0]) && q[1].compatible(v[1]) && k[0].compatible(v[0]) &&
                              k[1].compatible(v[1]) && k[3].compatible(q[3]),
                          "Q/K/V batch, tokens and key dimensions must agree");
    NODE_VALIDATION_CHECK(
        this, g.compatible(PartialShape{v[0], v[1], v[2], g[3]}) && (g[3].compatible(1) || g[3].compatible(q[3])),
        "Invalid scalar/vector gate");
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(5).compatible(PartialShape{v[0], v[1], v[2], 1}),
                          "Beta requires [B,T,Hv,1]");
    NODE_VALIDATION_CHECK(this, m_head_mapping == "tiled" || m_head_mapping == "grouped", "Unknown head mapping");
    NODE_VALIDATION_CHECK(this,
                          std::isfinite(m_q_l2_norm_eps) && m_q_l2_norm_eps > 0 && std::isfinite(m_k_l2_norm_eps) &&
                              m_k_l2_norm_eps > 0,
                          "Normalization epsilons must be finite and positive");
    for (auto heads : {q[2], k[2]})
        if (heads.is_static() && v[2].is_static())
            NODE_VALIDATION_CHECK(this, heads.get_length() > 0 && v[2].get_length() % heads.get_length() == 0,
                                  "Value heads must be divisible by Q/K heads");
    const auto state = bank_features(this, 3), pk = bank_features(this, 6), pv = bank_features(this, 7),
               pgb = bank_features(this, 8);
    NODE_VALIDATION_CHECK(this, state.compatible(v[2] * q[3] * v[3]), "State features require Hv*Dk*Dv");
    for (size_t port : {6, 7, 8})
        NODE_VALIDATION_CHECK(this, get_input_partial_shape(port)[2].compatible(get_input_partial_shape(3)[2]),
                              "Bank row capacities must agree");
    if (pk.is_static() && k[2].is_static() && k[3].is_static() && v[2].is_static() && v[3].is_static() &&
        g[3].is_static()) {
        const auto key_features = k[2].get_length() * k[3].get_length();
        NODE_VALIDATION_CHECK(
            this, key_features > 0 && pk.get_length() % key_features == 0 && pk.get_length() / key_features > 0,
            "Invalid pending key capacity");
        const auto pending = pk.get_length() / key_features;
        NODE_VALIDATION_CHECK(this, pv.compatible(v[2] * v[3] * pending) && pgb.compatible(v[2] * g[3] * (2 * pending)),
                              "Pending bank capacities must agree");
    }
    controls(this, 9, 10, 11, v[0]);
    set_output_type(0, get_input_element_type(0), v);
}
bool RecurrentGatedDeltaNetReplayUpdate::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("fuse_qk_l2norm", m_fuse_qk_l2norm);
    visitor.on_attribute("q_l2_norm_eps", m_q_l2_norm_eps);
    visitor.on_attribute("k_l2_norm_eps", m_k_l2_norm_eps);
    visitor.on_attribute("head_mapping", m_head_mapping);
    return true;
}
std::shared_ptr<Node> RecurrentGatedDeltaNetReplayUpdate::clone_with_new_inputs(const OutputVector& args) const {
    check_new_args_count(this, args);
    return std::make_shared<RecurrentGatedDeltaNetReplayUpdate>(args, m_fuse_qk_l2norm, m_q_l2_norm_eps,
                                                                m_k_l2_norm_eps, m_head_mapping);
}
} // namespace ov::intel_gpu::op
