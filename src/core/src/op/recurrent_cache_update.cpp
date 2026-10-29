// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "openvino/op/recurrent_cache_update.hpp"

#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/parameter.hpp"
#include <numeric>
namespace ov::op::internal {
namespace {
void validate_cache(const Node* op, const std::vector<size_t>& ports) {
    for (size_t i = 0; i < ports.size(); ++i) {
        const auto shape = op->get_input_partial_shape(ports[i]);
        NODE_VALIDATION_CHECK(op,
                              shape.rank().is_static() && shape.rank().get_length() == 4,
                              "Recurrent cache inputs require rank 4");
        const auto type = op->get_input_element_type(ports[i]);
        if (i == 0 || i == 6 || i == 8) {
            NODE_VALIDATION_CHECK(op, type == element::f16 || type == element::f32, "Cache/QKV requires f16 or f32");
        } else {
            NODE_VALIDATION_CHECK(op,
                                  type == element::i32 || type == element::i64,
                                  "Cache metadata requires i32 or i64");
        }
    }
    const auto cache = op->get_input_partial_shape(ports[0]);
    NODE_VALIDATION_CHECK(op, cache[0].compatible(1) && cache[1].compatible(1), "Cache requires [1,1,S,F]");
    for (size_t i : {1, 2}) {
        const auto shape = op->get_input_partial_shape(ports[i]);
        NODE_VALIDATION_CHECK(op,
                              shape[0].compatible(1) && shape[1].compatible(1) && shape[2].compatible(1),
                              "Row indices require [1,1,1,N]");
    }
    for (size_t i : {3, 4}) {
        NODE_VALIDATION_CHECK(op,
                              op->get_input_partial_shape(ports[i]).compatible(PartialShape{1, 1, 1, 1}),
                              "Offsets require [1,1,1,1]");
    }
    NODE_VALIDATION_CHECK(op,
                          op->get_input_partial_shape(ports[5]).compatible(PartialShape{1, 1, 1, 2}),
                          "Extra offsets require [destination, source begin]");
}
}  // namespace

RecurrentConvCacheUpdate::RecurrentConvCacheUpdate(const OutputVector& args, int64_t history)
    : Op(args),
      m_history(history) {
    constructor_validate_and_infer_types();
}
void RecurrentConvCacheUpdate::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 9, "Convolution cache requires 9 inputs");
    std::vector<size_t> ports(get_input_size());
    std::iota(ports.begin(), ports.end(), 0);
    validate_cache(this, ports);
    NODE_VALIDATION_CHECK(this, m_history > 0, "History must be positive");
    const auto qkv = get_input_partial_shape(6);
    NODE_VALIDATION_CHECK(this,
                          qkv[0].compatible(1) && qkv[1].compatible(get_input_partial_shape(1)[3]),
                          "QKV batch must match active rows");
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(0)[3].compatible(qkv[2] * m_history),
                          "Cache features must equal channels times history");
    const auto windows = get_input_partial_shape(7);
    NODE_VALIDATION_CHECK(this,
                          windows[0].compatible(1) && windows[1].compatible(1) && windows[3].compatible(2),
                          "Windows require [1,1,K,2]");
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(8).compatible(PartialShape{1, 1, qkv[2], m_history + 1}),
                          "Depthwise weights require [1,1,C,history+1]");
    set_output_type(0, get_input_element_type(6), PartialShape{1, qkv[1], qkv[3], qkv[2]});
    set_output_type(1, get_input_element_type(0), get_input_partial_shape(0));
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
    NODE_VALIDATION_CHECK(this, get_input_size() == 12, "Recurrent GDN requires 12 inputs");
    NODE_VALIDATION_CHECK(this, m_snapshot_count >= 1, "snapshot_count must be positive");
    const auto q = get_input_partial_shape(0), v = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(this, q.rank() == 4 && v.rank() == 4, "Q/V require rank 4");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0) == element::f16 || get_input_element_type(0) == element::f32,
                          "Recurrent GDN requires f16/f32 computation");
    for (size_t i : {1, 2, 4, 5})
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(i) == get_input_element_type(0),
                              "Q/K/V/gate/beta must use the same computation precision");
    auto state = std::make_shared<v0::Parameter>(get_input_element_type(0), PartialShape{q[0], v[2], q[3], v[3]});
    GatedDeltaNet base(
        OutputVector{input_value(0), input_value(1), input_value(2), state, input_value(4), input_value(5)},
        get_fuse_qk_l2norm(),
        get_q_l2_norm_eps(),
        get_k_l2_norm_eps());
    validate_cache(this, {3, 6, 7, 8, 9, 10});
    NODE_VALIDATION_CHECK(this, q[0].compatible(get_input_partial_shape(6)[3]), "Active row count must match batch");
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(3)[3].compatible(v[2] * q[3] * v[3]),
                          "Cache features must match H*K*V");
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(11).rank() == 4 &&
                              get_input_partial_shape(11).compatible(PartialShape{1, 1, 1, 4}),
                          "State update control requires [1,1,1,4]");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(11) == element::i32 || get_input_element_type(11) == element::i64,
                          "State update control requires i32/i64");
    set_output_size(2);
    set_output_type(0, get_input_element_type(0), base.get_output_partial_shape(0));
    set_output_type(1, get_input_element_type(3), get_input_partial_shape(3));
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
