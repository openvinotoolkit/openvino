// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/select.hpp"

#include <memory>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/validation_util.hpp"  // tbr
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/reference/select.hpp"
#include "select_shape_inference.hpp"

using namespace ngraph;

namespace ov {
namespace op {
namespace v1 {
Select::Select(const Output<Node>& arg0,
               const Output<Node>& arg1,
               const Output<Node>& arg2,
               const AutoBroadcastSpec& auto_broadcast)
    : Op({arg0, arg1, arg2}),
      m_auto_broadcast(auto_broadcast) {
    constructor_validate_and_infer_types();
}

void Select::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Select_validate_and_infer_types);
    // Condition element type check
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_dynamic() || get_input_element_type(0) == element::boolean,
                          "Argument 0 must have boolean element type (element type: ",
                          get_input_element_type(0),
                          ").");

    // Then/Else element type check
    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, get_input_element_type(1), get_input_element_type(2)),
                          "Argument 1 and 2 element types must match.");

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto input_shapes = get_node_input_partial_shapes(*this);
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, result_et, output_shapes[0]);
}

std::shared_ptr<Node> Select::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Select_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v1::Select>(new_args.at(0), new_args.at(1), new_args.at(2), m_auto_broadcast);
}

bool Select::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Select_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace detail {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorVector& output_values,
              const HostTensorVector& input_values,
              const AutoBroadcastSpec& autob) {
    using T = typename element_type_traits<ET>::value_type;

    const auto& in_cond = input_values[0];
    const auto& in_then = input_values[1];
    const auto& in_else = input_values[2];

    const auto& out = output_values[0];

    reference::select<T>(in_cond->get_data_ptr<char>(),
                         in_then->get_data_ptr<T>(),
                         in_else->get_data_ptr<T>(),
                         out->get_data_ptr<T>(),
                         in_cond->get_shape(),
                         in_then->get_shape(),
                         in_else->get_shape(),
                         autob);
    return true;
}

bool evaluate_select(const HostTensorVector& output_values,
                     const HostTensorVector& input_values,
                     const AutoBroadcastSpec& autob,
                     const element::Type_t& et) {
    bool rc = false;

    switch (et) {
        OPENVINO_TYPE_CASE(evaluate_select, i8, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, i16, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, i32, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, i64, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, u8, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, u16, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, u32, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, u64, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, bf16, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, f16, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, f32, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, f64, output_values, input_values, autob);
        OPENVINO_TYPE_CASE(evaluate_select, boolean, output_values, input_values, autob);
    default:
        rc = false;
        break;
    }

    return rc;
}
}  // namespace
}  // namespace detail

bool Select::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    OV_OP_SCOPE(v1_Select_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(validate_host_tensor_vector(input_values, 3));
    OPENVINO_ASSERT(validate_host_tensor_vector(output_values, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto autob = get_auto_broadcast();
    return detail::evaluate_select(output_values, input_values, autob, output_values[0]->get_element_type());
}

bool Select::evaluate_lower(TensorVector& output_values) const {
    return get_input_tensor(0).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool Select::evaluate_upper(TensorVector& output_values) const {
    return get_input_tensor(0).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool Select::has_evaluate() const {
    OV_OP_SCOPE(v1_Select_has_evaluate);
    switch (get_output_element_type(0)) {
    case element::boolean:
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}
}  // namespace v1
}  // namespace op
}  // namespace ov
