// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/select.hpp"

#include <memory>
#include <ngraph/validation_util.hpp>
#include <select_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/runtime/reference/select.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::Select);

op::v1::Select::Select(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       const Output<Node>& arg2,
                       const AutoBroadcastSpec& auto_broadcast)
    : Op({arg0, arg1, arg2}),
      m_auto_broadcast(auto_broadcast) {
    constructor_validate_and_infer_types();
}

void op::v1::Select::validate_and_infer_types() {
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

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    const std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0),
                                                        get_input_partial_shape(1),
                                                        get_input_partial_shape(2)};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, result_et, output_shapes[0]);
}

shared_ptr<Node> op::v1::Select::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Select_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Select>(new_args.at(0), new_args.at(1), new_args.at(2), m_auto_broadcast);
}

bool op::v1::Select::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Select_visit_attributes);
    visitor.on_attribute("auto_broadcast", m_auto_broadcast);
    return true;
}

namespace detail {
namespace {
template <element::Type_t ET>
bool evaluate(const HostTensorVector& output_values,
              const HostTensorVector& input_values,
              const op::AutoBroadcastSpec& autob) {
    using T = typename element_type_traits<ET>::value_type;

    const auto& in_cond = input_values[0];
    const auto& in_then = input_values[1];
    const auto& in_else = input_values[2];

    const auto& out = output_values[0];

    runtime::reference::select<T>(in_cond->get_data_ptr<char>(),
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
                     const op::AutoBroadcastSpec& autob,
                     const element::Type_t& et) {
    bool rc = false;

    switch (et) {
        NGRAPH_TYPE_CASE(evaluate_select, i8, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, i16, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, i32, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, i64, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, u8, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, u16, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, u32, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, u64, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, bf16, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, f16, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, f32, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, f64, output_values, input_values, autob);
        NGRAPH_TYPE_CASE(evaluate_select, boolean, output_values, input_values, autob);
    default:
        rc = false;
        break;
    }

    return rc;
}
}  // namespace
}  // namespace detail

bool op::v1::Select::evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const {
    OV_OP_SCOPE(v1_Select_evaluate);
    NGRAPH_CHECK(validate_host_tensor_vector(input_values, 3));
    NGRAPH_CHECK(validate_host_tensor_vector(output_values, 1));
    const auto autob = get_auto_broadcast();
    return detail::evaluate_select(output_values, input_values, autob, output_values[0]->get_element_type());
}

bool op::v1::Select::has_evaluate() const {
    OV_OP_SCOPE(v1_Select_has_evaluate);
    switch (get_output_element_type(0)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::bf16:
    case ngraph::element::f16:
    case ngraph::element::f32:
    case ngraph::element::f64:
    case ngraph::element::boolean:
        return true;
    default:
        break;
    }
    return false;
}
