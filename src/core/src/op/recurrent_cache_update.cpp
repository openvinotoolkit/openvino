// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "openvino/op/recurrent_cache_update.hpp"

#include "openvino/op/recurrent_cache.hpp"
namespace ov::op::internal {
RecurrentConvCacheUpdate::RecurrentConvCacheUpdate(const OutputVector& args, int64_t history)
    : Op(args),
      m_history(history) {
    constructor_validate_and_infer_types();
}
void RecurrentConvCacheUpdate::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 9, "RecurrentConvCacheUpdate requires nine inputs");
    // Reuse shape validation without extending the existing op API or deriving
    // from its implementation. GPU registration and execution are independent.
    RecurrentConvCache reference(input_values(), m_history);
    set_output_size(2);
    for (size_t i = 0; i < 2; ++i)
        set_output_type(i, reference.get_output_element_type(i), reference.get_output_partial_shape(i));
}
bool RecurrentConvCacheUpdate::visit_attributes(AttributeVisitor& v) {
    v.on_attribute("history", m_history);
    return true;
}
std::shared_ptr<Node> RecurrentConvCacheUpdate::clone_with_new_inputs(const OutputVector& a) const {
    check_new_args_count(this, a);
    return std::make_shared<RecurrentConvCacheUpdate>(a, m_history);
}
RecurrentGatedDeltaNetUpdate::RecurrentGatedDeltaNetUpdate(const OutputVector& args,
                                                           bool fuse_qk_l2norm,
                                                           float q_l2_norm_eps,
                                                           float k_l2_norm_eps,
                                                           int64_t snapshot_count)
    : Op(args),
      m_fuse_qk_l2norm(fuse_qk_l2norm),
      m_q_l2_norm_eps(q_l2_norm_eps),
      m_k_l2_norm_eps(k_l2_norm_eps),
      m_snapshot_count(snapshot_count) {
    constructor_validate_and_infer_types();
}
void RecurrentGatedDeltaNetUpdate::validate_and_infer_types() {
    RecurrentGatedDeltaNet reference(input_values(),
                                     m_fuse_qk_l2norm,
                                     m_q_l2_norm_eps,
                                     m_k_l2_norm_eps,
                                     m_snapshot_count);
    set_output_size(2);
    for (size_t i = 0; i < 2; ++i)
        set_output_type(i, reference.get_output_element_type(i), reference.get_output_partial_shape(i));
}
bool RecurrentGatedDeltaNetUpdate::visit_attributes(AttributeVisitor& v) {
    v.on_attribute("fuse_qk_l2norm", m_fuse_qk_l2norm);
    v.on_attribute("q_l2_norm_eps", m_q_l2_norm_eps);
    v.on_attribute("k_l2_norm_eps", m_k_l2_norm_eps);
    v.on_attribute("snapshot_count", m_snapshot_count);
    return true;
}
std::shared_ptr<Node> RecurrentGatedDeltaNetUpdate::clone_with_new_inputs(const OutputVector& a) const {
    check_new_args_count(this, a);
    return std::make_shared<RecurrentGatedDeltaNetUpdate>(a,
                                                          m_fuse_qk_l2norm,
                                                          m_q_l2_norm_eps,
                                                          m_k_l2_norm_eps,
                                                          m_snapshot_count);
}
}  // namespace ov::op::internal
