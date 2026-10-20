// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/op/recurrent_cache.hpp"

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
        if (i == 0 || i == 6) {
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
    : Op(args),
      m_key_size(key_size),
      m_value_size(value_size) {
    constructor_validate_and_infer_types();
}
void RecurrentStateGather::validate_and_infer_types() {
    validate_cache(this, 6);
    const auto cache = get_input_partial_shape(0);
    validate_matrix(this, m_key_size, m_value_size);
    set_output_type(0, get_input_element_type(0), PartialShape{1, 1, get_input_partial_shape(1)[3], cache[3]});
    set_output_type(1, get_input_element_type(0), cache);
}
bool RecurrentStateGather::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("key_size", m_key_size);
    visitor.on_attribute("value_size", m_value_size);
    return true;
}
std::shared_ptr<Node> RecurrentStateGather::clone_with_new_inputs(const OutputVector& args) const {
    check_new_args_count(this, args);
    return std::make_shared<RecurrentStateGather>(args, m_key_size, m_value_size);
}
RecurrentStateUpdate::RecurrentStateUpdate(const OutputVector& args, int64_t key_size, int64_t value_size)
    : Op(args),
      m_key_size(key_size),
      m_value_size(value_size) {
    constructor_validate_and_infer_types();
}
void RecurrentStateUpdate::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 3, "State update requires cache, states, control");
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
    validate_cache(this, 8);
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
    set_output_type(0, get_input_element_type(6), PartialShape{1, qkv[1], qkv[2], qkv[3] + m_history});
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
}  // namespace ov::op::internal
