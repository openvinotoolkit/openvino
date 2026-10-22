// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/recurrent_cache.hpp"

#include "openvino/op/parameter.hpp"
#include <limits>

#include "openvino/core/validation_util.hpp"

namespace ov::op::internal {
namespace {
void validate_cache(const Node* op, size_t inputs) {
    NODE_VALIDATION_CHECK(op, op->get_input_size() == inputs, "Unexpected recurrent cache input count");
    for (size_t i = 0; i < inputs; ++i) {
        const auto shape = op->get_input_partial_shape(i);
        NODE_VALIDATION_CHECK(op,
                              shape.rank().is_static() && shape.rank().get_length() == 4,
                              "Recurrent cache inputs require rank 4");
        const auto type = op->get_input_element_type(i);
        if (i == 0 || i == 6 || i == 8) {
            NODE_VALIDATION_CHECK(op, type == element::f16 || type == element::f32, "Cache/QKV requires f16 or f32");
        } else {
            NODE_VALIDATION_CHECK(op,
                                  type == element::i32 || type == element::i64,
                                  "Cache metadata requires i32 or i64");
        }
    }
    const auto cache = op->get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(op, cache[0].compatible(1) && cache[1].compatible(1), "Cache requires [1,1,S,F]");
    for (size_t i : {1, 2}) {
        const auto shape = op->get_input_partial_shape(i);
        NODE_VALIDATION_CHECK(op,
                              shape[0].compatible(1) && shape[1].compatible(1) && shape[2].compatible(1),
                              "Row indices require [1,1,1,N]");
    }
    for (size_t i : {3, 4}) {
        NODE_VALIDATION_CHECK(op,
                              op->get_input_partial_shape(i).compatible(PartialShape{1, 1, 1, 1}),
                              "Offsets require [1,1,1,1]");
    }
    NODE_VALIDATION_CHECK(op,
                          op->get_input_partial_shape(5).compatible(PartialShape{1, 1, 1, 2}),
                          "Extra offsets require [destination, source begin]");
}
void validate_matrix(const Node* op, int64_t key_size, int64_t value_size) {
    NODE_VALIDATION_CHECK(op,
                          (key_size == 0 && value_size == 0) || (key_size > 0 && value_size > 0),
                          "Both state matrix dimensions must be positive or zero");
    if (key_size > 0) {
        NODE_VALIDATION_CHECK(op,
                              key_size <= std::numeric_limits<int64_t>::max() / value_size,
                              "State matrix dimensions overflow");
        const auto feature = op->get_input_partial_shape(0)[3];
        NODE_VALIDATION_CHECK(op,
                              !feature.is_static() || feature.get_length() % (key_size * value_size) == 0,
                              "State matrix dimensions must divide cache features");
    }
}
}  // namespace
RecurrentStateGather::RecurrentStateGather(const OutputVector& args, int64_t key_size, int64_t value_size)
    : RecurrentStateGather(args, key_size, value_size, true) {}
RecurrentStateGather::RecurrentStateGather(const OutputVector& args,
                                           int64_t key_size,
                                           int64_t value_size,
                                           bool prepare_cache)
    : Op(args),
      m_key_size(key_size),
      m_value_size(value_size),
      m_prepare_cache(prepare_cache) {
    constructor_validate_and_infer_types();
}
void RecurrentStateGather::validate_and_infer_types() {
    validate_cache(this, 6);
    const auto cache = get_input_partial_shape(0);
    validate_matrix(this, m_key_size, m_value_size);
    set_output_size(m_prepare_cache ? 2 : 1);
    set_output_type(0, get_input_element_type(0), PartialShape{1, 1, get_input_partial_shape(1)[3], cache[3]});
    if (m_prepare_cache)
        set_output_type(1, get_input_element_type(0), cache);
}
bool RecurrentStateGather::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("prepare_cache", m_prepare_cache);
    visitor.on_attribute("key_size", m_key_size);
    visitor.on_attribute("value_size", m_value_size);
    return true;
}
std::shared_ptr<Node> RecurrentStateGather::clone_with_new_inputs(const OutputVector& args) const {
    check_new_args_count(this, args);
    return std::make_shared<RecurrentStateGather>(args, m_key_size, m_value_size, m_prepare_cache);
}
RecurrentStateUpdate::RecurrentStateUpdate(const OutputVector& args, int64_t key_size, int64_t value_size)
    : Op(args),
      m_key_size(key_size),
      m_value_size(value_size) {
    constructor_validate_and_infer_types();
}
void RecurrentStateUpdate::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 3 || get_input_size() == 7,
                          "State update requires cache, states, control and optional reset/reorder metadata");
    for (size_t i = 0; i < 2; ++i) {
        const auto shape = get_input_partial_shape(i);
        const auto type = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              shape.rank().is_static() && shape.rank().get_length() == 4 && shape[0].compatible(1) &&
                                  shape[1].compatible(1),
                              "State update requires [1,1,N,F]");
        NODE_VALIDATION_CHECK(this, type == element::f16 || type == element::f32, "States require f16/f32");
    }
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(0)[3].compatible(get_input_partial_shape(1)[3]),
                          "State update feature sizes must match");
    const auto control = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(
        this,
        control.rank().is_static() && control.rank().get_length() == 4 && control.compatible(PartialShape{1, 1, 1, 4}),
        "State update control requires four values");
    const auto type = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this, type == element::i32 || type == element::i64, "Control requires i32/i64");
    validate_matrix(this, m_key_size, m_value_size);
    if (get_input_size() == 7) {
        for (size_t i = 3; i < 7; ++i) {
            const auto type = get_input_element_type(i);
            NODE_VALIDATION_CHECK(this,
                                  type == element::i32 || type == element::i64,
                                  "Reset/reorder metadata requires i32/i64");
            const PartialShape expected =
                i == 3 ? PartialShape{1, 1, 1, -1} : (i == 6 ? PartialShape{1, 1, 1, 2} : PartialShape{1, 1, 1, 1});
            NODE_VALIDATION_CHECK(
                this,
                get_input_partial_shape(i).rank() == 4 && get_input_partial_shape(i).compatible(expected),
                "Invalid reset/reorder metadata shape");
        }
    }
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
bool RecurrentStateUpdate::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("key_size", m_key_size);
    visitor.on_attribute("value_size", m_value_size);
    return true;
}
std::shared_ptr<Node> RecurrentStateUpdate::clone_with_new_inputs(const OutputVector& args) const {
    check_new_args_count(this, args);
    return std::make_shared<RecurrentStateUpdate>(args, m_key_size, m_value_size);
}
RecurrentConvCache::RecurrentConvCache(const OutputVector& args, int64_t history) : Op(args), m_history(history) {
    constructor_validate_and_infer_types();
}
void RecurrentConvCache::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 8 || get_input_size() == 9,
                          "Convolution cache requires 8 or 9 inputs");
    // Input 8 optionally supplies depthwise weights to fuse the arithmetic.
    validate_cache(this, get_input_size());
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
    if (get_input_size() == 9) {
        NODE_VALIDATION_CHECK(this,
                              get_input_partial_shape(8).compatible(PartialShape{1, 1, qkv[2], m_history + 1}),
                              "Depthwise weights require [1,1,C,history+1]");
    }
    set_output_type(0,
                    get_input_element_type(6),
                    get_input_size() == 9 ? PartialShape{1, qkv[1], qkv[3], qkv[2]}
                                          : PartialShape{1, qkv[1], qkv[2], qkv[3] + m_history});
    set_output_type(1, get_input_element_type(0), get_input_partial_shape(0));
}
bool RecurrentConvCache::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("history", m_history);
    return true;
}
std::shared_ptr<Node> RecurrentConvCache::clone_with_new_inputs(const OutputVector& args) const {
    check_new_args_count(this, args);
    return std::make_shared<RecurrentConvCache>(args, m_history);
}
RecurrentGatedDeltaNet::RecurrentGatedDeltaNet(const OutputVector& args, bool norm, float qe, float ke, int64_t count) {
    set_arguments(args);
    initialize_attributes(norm, qe, ke, count);
    constructor_validate_and_infer_types();
}
void RecurrentGatedDeltaNet::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 12, "Recurrent GDN requires 12 inputs");
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
        get_k_l2_norm_eps(),
        get_snapshot_count());
    RecurrentStateGather gather(
        OutputVector{input_value(3), input_value(6), input_value(7), input_value(8), input_value(9), input_value(10)});
    NODE_VALIDATION_CHECK(this, q[0].compatible(get_input_partial_shape(6)[3]), "Active row count must match batch");
    NODE_VALIDATION_CHECK(this,
                          get_input_partial_shape(3)[3].compatible(v[2] * q[3] * v[3]),
                          "Cache features must match H*K*V");
    auto flat = std::make_shared<v0::Parameter>(get_input_element_type(0),
                                                PartialShape{1, 1, -1, get_input_partial_shape(3)[3]});
    RecurrentStateUpdate update(OutputVector{input_value(3), flat, input_value(11)});
    set_output_size(2);
    set_output_type(0, get_input_element_type(0), base.get_output_partial_shape(0));
    set_output_type(1, get_input_element_type(3), get_input_partial_shape(3));
}
std::shared_ptr<Node> RecurrentGatedDeltaNet::clone_with_new_inputs(const OutputVector& args) const {
    return std::make_shared<RecurrentGatedDeltaNet>(args,
                                                    get_fuse_qk_l2norm(),
                                                    get_q_l2_norm_eps(),
                                                    get_k_l2_norm_eps(),
                                                    get_snapshot_count());
}
}  // namespace ov::op::internal
